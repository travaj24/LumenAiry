"""PMM shared core: GLL/Lagrange spectral-element basis, the metric /
convection / covariant slant generators, the slant solvers + stabilize selector,
S-matrix cascade, and the far-field nodal->Rayleigh projection.
NOT a public import surface -- use ``lumenairy.elements.pmm``.

CONVENTIONS shared with the RCWA family (order-count kwarg spellings, the
``formulation`` / ``stabilize`` / ``symmetry`` defaults that deliberately differ
between siblings, and the single ``Re(kz) > 0`` propagating-order mask): see the
CROSS-FAMILY KWARG / DEFAULT MAP in the module docstring of
``lumenairy/elements/rcwa/_core.py`` (audit M7 2026-07-25).  PMM's ``degree`` is
a spectral-element POLYNOMIAL degree and is NOT interchangeable with an RCWA
retained-harmonic ``n_orders``; PMM's ``n_orders`` is an alias of
``far_field_orders`` (the projected Rayleigh-order count) on the 1-D entry
points."""
from __future__ import annotations

import functools
import threading
import warnings
from collections import OrderedDict

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import Legendre

# Backend detection for the JAX (differentiable) dispatch in pmm_efficiency_1d.
# Mirrors rcwa's pattern: a JAX input routes to the self-contained jnp twin,
# while a NumPy input falls through to the original (byte-identical) code.
from ...backend import is_jax_array

# Reused for the slanted-grating solver: the slant breaks the +/-q field
# symmetry (like a full-3x3 tensor layer), so it needs the GENERALIZED
# (explicit forward/backward) S-matrix.  rcwa does NOT import pmm, so this
# top-level import introduces no cycle.
from ..rcwa import (
    _interface_smatrix_general,
    _propagation_star,
    _propagation_star_general,
    # F1 (PMM_RCWA_AUDIT_2026_07_02): the Redheffer/propagation-star algebra is
    # convention-free, so share RCWA's structure-aware copies instead of keeping
    # a performance-diverged local fork.  ``_redheffer_star`` gains the zero-block
    # shortcut (skips two dense 2N inverses against a propagation S-matrix's zero
    # diagonal blocks -- proven bit-identical); ``_propagation_star(_general)``
    # collapses the whole star against a propagation S-matrix to row/col scaling.
    _redheffer_star,
)

# Incidence-medium guard shared with the 2-D PMM paths (twod.py / stack2d.py):
# the rcwa guard expects the INTERNAL exp(+i w t) eps convention, so PUBLIC-
# convention callers pass np.conj(eps_sup) (mirrors twod_staggered's bridge).
from ..rcwa._core import _require_propagating_incidence

_C = np.complex128

# Minimum slant (radians) for the covariant oblique-coordinate path.  The covariant
# frame ``u = x - tan(phi) z`` DEGENERATES as ``phi -> 0`` (it becomes the identity;
# the TE/TM eigenvalues collapse to exactly degenerate and the interface mode-match
# goes near-singular, a ~1e8-amplified inversion).  At the actual eigenvalues the
# result is correct, but with a ~0-gap the eigenVECTORS are maximally sensitive to
# the last-bit BLAS rounding, so different LAPACK builds land on opposite sides of
# the instability (e.g. OpenBLAS on CI returned a blown-up R+T while MKL did not).
# Below this angle the grating is ~vertical, where the convection treatment is BOTH
# exact AND well-conditioned, so 'auto' and an explicit 'covariant' both route there.
# 1e-3 rad ~ 0.057deg -- far below any slant where the spectral covariant win matters.
_COV_MIN_SLANT_RAD = 1.0e-3


def _resolve_order_count(far_field_orders, n_orders):
    """Cross-suite alias: accept ``n_orders`` (the RCWA / 2-D PMM spelling) as a
    synonym for ``far_field_orders`` (the historical 1-D PMM spelling) on every
    public 1-D entry point.  ``n_orders`` overrides when supplied; ``None`` (the
    default) keeps ``far_field_orders``.  Downstream code still coerces with
    ``int(...)`` so no coercion happens here.

    NB (audit P3, 2026-06-09): the "alias wins, no equality check" rule is
    DELIBERATE -- it is the cross-suite drop-in-substitution contract established
    by audit F2/F3 (a config carrying BOTH spellings resolves identically in
    every suite), and is pinned by ``test_v5_12_0_naming_aliases``.  Adding a
    raise-on-mismatch here was considered and REJECTED: it would break that
    intentional, tested feature.  Pass only one spelling in practice."""
    return far_field_orders if n_orders is None else n_orders



def _promote_eps_tensor(eps):
    """Promote a scalar (isotropic) permittivity to a ``(3, 3)`` tensor; pass a
    ``(3, 3)`` tensor through unchanged.  Mirrors ``PMMStack._as_tensor`` so the
    functional :func:`pmm_1d` dispatcher honours the documented "scalar promoted
    to an isotropic tensor" contract -- for BOTH NumPy and JAX scalars.  A 0-d
    JAX scalar is promoted to ``eps * jnp.eye(3)``, which stays differentiable
    (the gradient flows to the index value); a JAX array that is already a tensor
    is passed through unchanged."""
    if is_jax_array(eps):
        if getattr(eps, "ndim", 2) == 0:
            import jax.numpy as _jnp
            return eps * _jnp.eye(3, dtype=_jnp.complex128)
        return eps
    M = np.asarray(eps, dtype=_C)
    if M.ndim == 0:
        M = M * np.eye(3, dtype=_C)
    return M



def _resolve_incidence(angle, theta):
    """Cross-dimension alias: accept ``theta`` (the 2-D / conical polar-angle
    spelling, also used by ``RCWAStack``) as a synonym for ``angle`` (the 1-D
    classical-mount incidence angle).  ``theta`` IS ``angle`` -- the SAME number,
    NO scaling or conversion, both measured from the ``+z`` surface normal; the
    1-D mount is planar (azimuth ``phi = 0``).  ``theta`` overrides when supplied;
    ``None`` (the default) keeps ``angle``.

    NB (audit P3, 2026-06-09): the "theta wins, no equality check" rule is
    DELIBERATE -- it is the cross-suite drop-in-substitution contract established
    by audit F2/F3 (``set_source(angle=A, theta=T)`` resolves to ``T`` in EVERY
    suite), pinned by ``test_v5_12_0_naming_aliases``.  Adding a raise-on-mismatch
    here was considered and REJECTED: it would break that intentional, tested
    feature.  Pass only one spelling in practice."""
    return angle if theta is None else theta


def _resolve_incidence_checked(fn_name, angle, theta):
    """Resolve the ``angle``/``theta`` alias, then REJECT back-side incidence
    (audit P3-29): the angle enters the 1-D solve only as ``kx0 ~ sin(angle)``,
    so ``|angle| >= pi/2`` would otherwise alias BYTE-IDENTICALLY to the
    supplementary front-side angle ``pi - angle`` -- a plausible,
    energy-conserving answer for the WRONG geometry.  Raises ``ValueError``
    instead.  Composes with (does not duplicate) the grazing guard, which
    catches ``kz_inc ~ 0`` just BELOW ``pi/2``.  A TRACED JAX angle (under
    ``jit``/``grad``) has no concrete value to range-check and SKIPS the guard
    (the rcwa ``_reject_jax_offplane`` tracer carve-out); a CONCRETE JAX angle
    is checked.  A non-numeric angle is passed through for the solver's own
    coercion to raise on.

    This is the ONE shared checked resolver for the whole PMM suite: the 1-D
    entry points in :mod:`.oned` and the ``PMMStack`` source setters
    (:meth:`PMMStack.set_source` etc.) route through it, so a back-side angle
    is rejected identically everywhere (audit S1-7: ``set_source`` formerly
    bypassed this and silently solved the supplementary front-side geometry)."""
    angle = _resolve_incidence(angle, theta)
    if is_jax_array(angle):
        try:                                 # concrete JAX array -> inspectable
            a_c = float(np.asarray(angle))
        except Exception:                    # tracer -> not materialisable
            return angle
    else:
        try:
            a_c = float(angle)
        except (TypeError, ValueError):      # non-numeric: solver coercion raises
            return angle
    if not abs(a_c) < 0.5 * np.pi:           # NaN also fails this comparison
        raise ValueError(
            f"{fn_name}: incidence angle must satisfy |angle| < pi/2 "
            f"(front-side illumination, measured from the +z surface normal); "
            f"got {a_c} rad.  A past-grazing angle would silently alias to "
            f"the supplementary front-side angle (pi - angle).")
    return angle



# ``stabilize`` robust-selection parameters.  PMM has two distinct off-curve
# failure modes vs polynomial degree: (1) discrete RESONANCES at isolated degrees
# that INFLATE sum(R)+sum(T) above the converged value (sparse at low degree,
# proliferating into multi-degree bands at high degree), and (2) UNDER-
# CONVERGENCE at low degree on high-index gratings, where the total sits BELOW
# the converged value (a power deficit).  The clean answer is therefore neither
# the maximum nor the minimum power but the CONSENSUS that the converged degrees
# agree on.  The selector scans upward, collects the PASSIVE solves (total within
# ``_PASSIVE_TOL`` of unity -- discards the super-unity resonances), and once
# ``_MIN_CLUSTER`` of them agree within ``_CLUSTER_TOL`` (the converged plateau)
# returns the requested degree if its own total is in that cluster (degree / DOF
# preserved in the common clean case), else the nearest clean degree.  Both a
# lone low outlier (under-convergence) and a lone high outlier (resonance) are
# rejected as non-consensus.
_STABILIZE_MAX_SCAN = 16    # hard cap on degrees scanned (covers >=7-wide bands)
# PER-ORDER corroboration width: TWO passive degrees that agree per-order are
# strong evidence of convergence (the per-order signature is a far stronger
# signal than the total power, and a resonance-contaminated passive degree is
# isolated -- it has no per-order partner).  A heavily-resonant config can leave
# only two clean degrees in the window before the next resonance, so requiring 3
# would miss them and fall back to a contaminated degree.
_MIN_PLATEAU = 2
_PASSIVE_TOL = 1.0e-3       # reject super-unity resonances; accept lossless R+T=1
# converged degrees agree to ~1e-6 and differ across the window only by the
# convergence drift (~1e-4); resonances inflate, and under-convergence deflates,
# the power by >~2e-3.  This width cleanly separates the converged cluster from
# both off-curve modes and bounds the worst-case efficiency error of a returned
# degree.
_CLUSTER_TOL = 5.0e-4
# PER-ORDER convergence tolerance for the consensus.  The total-power gate alone
# is blind to under-convergence (the S-matrix conserves sum(R)+sum(T) even when
# the modal basis is under-resolved and the per-order split is wrong by tens of
# percent -- a silent error for high-index / many-order gratings at the default
# degree).  Two solves join the converged plateau only if their PER-ORDER
# efficiencies (and the Jones matrix) also agree to this width; an under-resolved
# degree fails to match the higher-degree plateau and is excluded (the consensus
# then returns a converged degree, or warns/raises if none is found).
_PER_ORDER_TOL = 3.0e-3


def _aligned_max_diff(rec_a, rec_b):
    """Max absolute PER-ORDER efficiency difference between two scanned solves,
    aligned by integer Rayleigh order (an order present in only one solve counts
    at its full magnitude), plus the ``|Jones|`` difference when both carry one.

    ``rec = (orders, effs, jones_or_None)`` where ``effs`` is a tuple of arrays
    whose LAST axis is the order (so ``A[..., i]`` is order ``i`` -- works for the
    scalar ``(N,)`` R/T and the Jones ``(2, N)`` R/T alike).  This is the
    convergence signal the total power alone misses: the S-matrix conserves
    sum(R)+sum(T) even when the polynomial basis is under-resolved and the
    per-order split is still moving."""
    oa, effs_a, Ja = rec_a
    ob, effs_b, Jb = rec_b

    def _key(o):
        # 1-D orders are scalars; 2-D orders are (m, n) pairs -- both hashable
        a = np.atleast_1d(np.asarray(o))
        return int(a[0]) if a.size == 1 else tuple(int(v) for v in a)

    ia = {_key(o): i for i, o in enumerate(oa)}
    ib = {_key(o): i for i, o in enumerate(ob)}
    d = 0.0
    for k in set(ia) | set(ib):
        for A, B in zip(effs_a, effs_b):
            va = A[..., ia[k]] if k in ia else 0.0
            vb = B[..., ib[k]] if k in ib else 0.0
            d = max(d, float(np.max(np.abs(np.asarray(va) - np.asarray(vb)))))
    if Ja is not None and Jb is not None:
        d = max(d, float(np.max(np.abs(np.asarray(Ja) - np.asarray(Jb)))))
    return d



def _converged_cluster(records, passive, tol, min_cluster):
    """Indices of the largest group of PASSIVE solves that MUTUALLY (pairwise)
    agree PER-ORDER (and on the Jones matrix) within ``tol`` -- the converged
    plateau.

    ``records[i]`` is the ``rec`` tuple for :func:`_aligned_max_diff`; ``passive``
    is the aligned bool list.  Returns the index list (sorted) when it reaches
    ``min_cluster`` members, else ``[]``.  Clustering on the per-order signature
    (not the total) is what rejects an under-converged-but-energy-passive solve:
    such a degree fails to agree with the higher-degree plateau and is excluded.

    The group is a CLIQUE, grown greedily around each anchor (audit P3-26: the
    previous anchor-star admitted members up to ``2*tol`` apart, quietly
    doubling the documented worst-case spread of the consensus pick); on a
    genuinely converged plateau -- every pair within ``tol`` -- the result is
    identical to the old star.
    """
    pidx = [i for i, p in enumerate(passive) if p]
    best = []
    for a in pidx:
        grp = [a]
        for b in pidx:
            if b != a and all(
                    _aligned_max_diff(records[b], records[m]) <= tol
                    for m in grp):
                grp.append(b)
        if len(grp) > len(best):
            best = grp
    return sorted(best) if len(best) >= min_cluster else []



# ===========================================================================
# Subsectional Gauss-Lobatto-Legendre (GLL) spectral-element primitives
# ===========================================================================

# ===========================================================================
# PERFORMANCE NOTES
# ---------------------------------------------------------------------------
# The dominant cost of a PMM solve is the dense ``np.linalg.eig`` on the layer
# generator (~85% of runtime), which depends on (eps, k0, slant) and so changes
# every solve AND every stabilize-scan degree -- it is fundamentally per-solve
# and is NOT cached.  The single biggest speed lever is therefore ACCURACY-PER-
# DEGREE, not caching: an in-plane slanted cell now defaults to the SPECTRAL
# covariant factorization (``'auto'``), reaching matched accuracy in ~100-2400x
# fewer degrees (hence a far smaller eig) than the algebraic convection path.
#
# What IS memoized here is only the GEOMETRY-ONLY reference machinery -- the GLL
# nodes/weights (Legendre root-find) and the barycentric differentiation matrix
# (O(n^2) Python loop).  These are pure functions of ``degree`` alone, rebuilt on
# every grid build, so caching them is free, exact (integer / node-coordinate
# keys, no float-physics staleness), and helps fixed-geometry sweeps and the
# stabilize scan.  Deliberately DEFERRED as risky-for-marginal-gain (see the
# fdtd/pmm roadmap): an analytic (eig-free) homogeneous half-space basis, a
# k0-independent generator-block precompute + float-keyed layer-mode cache, and a
# sparse-degree stabilize probe (which could declare a false plateau on the
# load-bearing convergence selector).  For production fixed-geometry sweeps,
# ``stabilize=False`` at a once-validated degree is the supported fast path.
# ===========================================================================
def _readonly(*arrays):
    """Mark arrays read-only (cache-poisoning guard) and return them."""
    for a in arrays:
        a.setflags(write=False)
    return arrays if len(arrays) > 1 else arrays[0]



@functools.lru_cache(maxsize=64)
def _gll_nodes_weights(degree: int):
    """GLL nodes (``degree + 1`` of them) and quadrature weights on ``[-1, 1]``:
    endpoints +/-1 plus the roots of ``P_degree'``; ``w_i = 2 / (degree
    (degree+1) P_degree(x_i)^2)``.

    Memoized on ``degree`` (PERF): a pure geometry function whose Legendre
    root-find recurs on EVERY grid build -- every solve, every stabilize-scan
    degree, every step of a fixed-geometry wavelength/angle sweep.  The cached
    arrays are returned READ-ONLY so an accidental in-place write raises instead
    of silently poisoning the cache (callers only ever map/scale them into new
    arrays: ``ref_w * J``, physical-coordinate maps)."""
    if degree == 1:
        return _readonly(np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
    interior = np.sort(Legendre.basis(degree).deriv().roots().real)
    nodes = np.concatenate([[-1.0], interior, [1.0]])
    PD = Legendre.basis(degree)
    w = 2.0 / (degree * (degree + 1) * (PD(nodes) ** 2))
    return _readonly(nodes, w)



# Memo for the barycentric differentiation matrix, keyed on the node coordinates
# (one entry per distinct GLL degree -- a small bounded set in practice).  Guarded
# by a companion lock for safe concurrent reader-writer access (library cache
# policy), and enrolled with the central cache registry at the bottom of the module.
_LAGRANGE_DREF_CACHE: dict = {}
_LAGRANGE_DREF_LOCK = threading.Lock()


def _clear_pmm_caches() -> None:
    """Clear the PMM module-level caches (enrolled with the library cache
    registry, so the global 'clear all caches' path empties them too)."""
    with _LAGRANGE_DREF_LOCK:
        _LAGRANGE_DREF_CACHE.clear()
    _clear_geo_eig_cache()



def _lagrange_derivative_matrix(nodes):
    """Differentiation matrix ``D[i, j] = l_j'(x_i)`` for the nodal Lagrange
    basis (barycentric form).

    Memoized on the node coordinates (PERF): the ``O(n^2)`` Python barycentric
    build recurs on every grid build for the same degree.  Same read-only
    poisoning-guard as :func:`_gll_nodes_weights`; ``nodes`` here is always the
    cached GLL array, so the ``tobytes`` key is one entry per degree."""
    key = nodes.tobytes()
    with _LAGRANGE_DREF_LOCK:
        cached = _LAGRANGE_DREF_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(nodes)
    Dmat = np.zeros((n, n))
    wb = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                wb[j] /= (nodes[j] - nodes[k])
    for i in range(n):
        for j in range(n):
            if i != j:
                Dmat[i, j] = (wb[j] / wb[i]) / (nodes[i] - nodes[j])
        Dmat[i, i] = -np.sum([Dmat[i, k] for k in range(n) if k != i])
    Dmat = _readonly(Dmat)
    with _LAGRANGE_DREF_LOCK:
        _LAGRANGE_DREF_CACHE[key] = Dmat
    return Dmat



def _graded_boundaries(a: float, b: float, n_el: int, grade: bool):
    """Element boundaries on ``[a, b]``.  ``grade=False`` -> uniform;
    ``grade=True`` -> Chebyshev-Lobatto clustering toward BOTH ends (the walls),
    which resolves the metal-corner field singularity (the TM bottleneck)."""
    if n_el <= 1:
        return np.array([a, b])
    if not grade:
        return np.linspace(a, b, n_el + 1)
    i = np.arange(n_el + 1)
    s = 0.5 * (1.0 - np.cos(np.pi * i / n_el))      # clustered at 0 and 1
    return a + (b - a) * s



def _build_sem(period, d_wall, eps_ridge, eps_groove, degree,
               n_ridge_el, n_groove_el, grade):
    """Assemble the periodic C0 spectral-element Galerkin operators
    (``S0, Peps, Pinv, L, Linv``) on the global nodal basis.

    Elements: ``n_ridge_el`` (eps_ridge) spanning ``[0, d_wall]`` then
    ``n_groove_el`` (eps_groove) spanning ``[d_wall, period]``, each degree
    ``degree``, optionally graded toward the walls.  Element boundaries land on
    both grating walls, so ``eps`` is constant per element and every integral is
    exact by GLL quadrature.  Returns a dict of matrices + the local->global
    map.  Periodicity (normal incidence, Bloch phase 1) shares the last global
    node with the first.
    """
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    rb = _graded_boundaries(0.0, d_wall, n_ridge_el, grade)
    gb = _graded_boundaries(d_wall, period, n_groove_el, grade)
    elem_bnds = (list(zip(rb[:-1], rb[1:], [eps_ridge] * n_ridge_el))
                 + list(zip(gb[:-1], gb[1:], [eps_groove] * n_groove_el)))
    n_el = len(elem_bnds)

    # global node numbering: shared element-boundary nodes (C0); the very last
    # node wraps onto global node 0 (periodic).
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]      # share with previous element
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0                            # periodic wrap
    n_glob = last

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    S0, Peps, Pinv, L, Linv = _z(), _z(), _z(), _z(), _z()
    C, Cinv = _z(), _z()                            # convection (oblique only)
    for e in range(n_el):
        xl, xr, eps = elem_bnds[e]
        J = 0.5 * (xr - xl)                         # dx/dxi
        inv = 1.0 / eps
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)                         # GLL mass (diagonal)
        Kloc = (Dphys.T * wel) @ Dphys              # stiffness
        Cloc = Mloc @ Dphys                         # INT phi_i phi_j' (convection)
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        S0[ix] += Mloc
        Peps[ix] += eps * Mloc
        Pinv[ix] += inv * Mloc
        L[ix] += Kloc
        Linv[ix] += inv * Kloc
        C[ix] += Cloc
        Cinv[ix] += inv * Cloc
    return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, C=C, Cinv=Cinv,
                n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds, degree=degree,
                ref_nodes=ref_nodes)



def _ill_scaled(A, ratio=1e-12):
    """``True`` when ``A`` is element-size (sliver-element) ill-scaled: the
    spectral-element mass/stiffness carry the element Jacobian ``J`` on their
    diagonal, so a grid spanning a huge width ratio (e.g. a 1 nm liner next to a
    500 nm region, or near-coincident walls in a tapered stack's union grid) gives
    ``min|diag| << max|diag|`` and ``cond ~ w_max/w_min``.  The threshold is
    deliberately generous (``cond >~ 1e12``) so that EVERY well-scaled geometry in
    normal use returns ``False`` and takes the plain, bit-identical path; only the
    genuinely pathological thin-feature grids are equilibrated."""
    d = np.abs(np.diag(A))
    dmax = float(d.max()) if d.size else 0.0
    return dmax > 0.0 and float(d.min()) < ratio * dmax



def _equil_scale(A):
    """Symmetric (Jacobi) equilibration scaling ``d_i = 1/sqrt(|A_ii|)`` with a
    floor on near-zero diagonals (a genuinely zero-width / ``J=0`` element, which
    only the geometry-side wall-merge can truly remove)."""
    d = np.sqrt(np.abs(np.diag(A)))
    d = np.maximum(d, d.max() * 1e-13)
    return 1.0 / d



def _safe_inv(A):
    """``inv(A)`` with symmetric Jacobi equilibration when ``A`` is element-size
    ill-scaled.  Equilibration is the EXACT identity ``inv(A) = D inv(D A D) D``
    (``D = diag(1/sqrt(diag A))``) for a real-positive diagonal (the SE mass
    ``S0``), and a conditioning-reducing similarity rescale otherwise: the matrix
    actually inverted has unit diagonal (``cond ~ degree^2`` instead of
    ``w_max/w_min``).  Well-scaled ``A`` -> plain ``inv`` (BIT-IDENTICAL)."""
    if not _ill_scaled(A):
        return np.linalg.inv(A)
    di = _equil_scale(A)
    return di[:, None] * np.linalg.inv((di[:, None] * A) * di[None, :]) * di[None, :]



def _safe_solve(A, B):
    """``solve(A, B)`` with the same equilibration gate as :func:`_safe_inv`
    (``A^-1 B = D (D A D)^-1 D B``).  Well-scaled ``A`` -> plain ``solve``."""
    if not _ill_scaled(A):
        return np.linalg.solve(A, B)
    di = _equil_scale(A)
    return di[:, None] * np.linalg.solve((di[:, None] * A) * di[None, :],
                                         di[:, None] * B)



def _safe_geig(A, B):
    """Generalized eig ``sla.eig(A, B)`` with symmetric equilibration of the pencil
    when the mass ``B`` is element-size ill-scaled.  The congruence ``D(.)D``
    leaves the eigenvalues invariant (``A x = q^2 B x`` <=> ``DAD z = q^2 DBD z``
    with ``x = D z``), so this only rescales the conditioning.  Well-scaled ``B``
    -> plain ``sla.eig`` (BIT-IDENTICAL)."""
    if not _ill_scaled(B):
        return sla.eig(A, B)
    di = _equil_scale(B)
    q2, z = sla.eig((di[:, None] * A) * di[None, :], (di[:, None] * B) * di[None, :])
    return q2, di[:, None] * z


def _fast_geig(A, B):
    """FOLDED standard eigensolve ``eig(B^-1 A)`` -- the generalized modal pencil
    ``A x = q^2 B x`` solved as a standard problem.  ~1.5-2x faster than the QZ
    :func:`_safe_geig` (no generalized Schur), and the dominant cost of a PMM
    solve is this dense eig.

    Valid because ``B`` is the nodal MASS (``S0`` for TE, ``Pinv`` for TM) --
    well-conditioned by construction (cf. the in-code note "B is well-conditioned
    here").  Same equilibration gate as ``_safe_geig`` (so ill-SCALED element
    sizes are handled identically); an LU-pivot ratio guard falls back to the
    robust generalized QZ if ``B`` is near-SINGULAR (e.g. an extreme-``eps``
    metal corner), so the speed-up never trades away physical accuracy.  The fold
    reproduces the ``_safe_geig`` spectrum to ~1e-12 -- the JAX twin folds
    ``eig(solve(B, A))`` identically and is validated to that tolerance."""
    if _ill_scaled(B):
        di = _equil_scale(B)
        Ae = (di[:, None] * A) * di[None, :]
        Be = (di[:, None] * B) * di[None, :]
    else:
        di = None
        Ae, Be = A, B
    lu, piv = sla.lu_factor(Be)
    du = np.abs(np.diag(lu))
    if du.size and du.min() <= 1e-12 * du.max():       # near-singular -> robust QZ
        return _safe_geig(A, B)
    q2, z = sla.eig(sla.lu_solve((lu, piv), Ae))
    return (q2, z) if di is None else (q2, di[:, None] * z)


def _forward_branch_flip(q, xp=np):
    """Sign-select each modal ``q`` onto the FORWARD (forward-decaying / +z)
    branch -- the noise-robust selector formerly COPY-PASTED verbatim across the
    five scalar-vertical PMM generator sites (audit S1-8: the exact multi-copy
    pattern that bred the six-copy factor-i defect).

    For lossless media the eigen-operator is (near-)Hermitian, so the QZ eig
    leaks ~1e-15 imaginary noise; a naive ``q.imag < 0`` sign test would flip
    near-real (propagating) modes on that noise and spawn dense spurious
    resonances (the v5.14 robustness-audit P1).  Flip ONLY when clearly
    backward: ``Im(q) < -tol`` (evanescent, backward-growing) or, inside the
    near-real band ``|Im(q)| <= tol``, ``Re(q) < 0``.  ``tol`` is relative to the
    largest ``|q|`` (floored at 1.0), so the guard scales with the mode spectrum.

    ``xp`` selects the array module: NumPy (default) materialises ``tol`` as a
    concrete Python float via ``max(float(...), 1.0)``, exactly as the historical
    NumPy copies did; passing ``jax.numpy`` keeps ``tol`` traced through
    ``jnp.maximum`` so the derivative w.r.t. the incidence angle still flows --
    reproducing the former JAX copy byte-for-byte.  This is a pure consolidation:
    every routed call site produces bit-identical output."""
    if xp is np:
        tol = 1e-8 * max(float(np.max(np.abs(q))), 1.0)
    else:
        tol = 1e-8 * xp.maximum(xp.max(xp.abs(q)), 1.0)
    flip = (q.imag < -tol) | ((xp.abs(q.imag) <= tol) & (q.real < 0.0))
    return xp.where(flip, -q, q)



def _freeze_cached(obj):
    """Mark every ndarray reachable in a cached value NON-WRITEABLE, in place.

    W7 A13 (2026-07-26).  The PMM caches hand out the STORED objects by
    IDENTITY, so a caller that writes into one silently poisons every later
    solve that hits the same key.  Measured pre-fix (mutate one entry by
    ``+= 1e-3``, then re-solve): ``PMM2DStackHybrid._geom_cache`` 21 of 23
    arrays writeable -> next solve drifts 1.543e-06;
    ``_PreparedPMMStack._eig_cache`` 12 of 12 -> 7.844e-07;
    ``stack2d._epsF_cache`` -> ``internal_field`` Ez drifts 1.377e-04; and
    ``_jax_twod._STATIC_CACHE`` 8 of 8 at MODULE scope, so the poison
    survives for the whole process.  No in-tree caller mutates them, so this
    closes a hardening gap rather than a live defect -- the same treatment
    ``_cached_geo_eig`` (M9), ``_LAGRANGE_DREF_CACHE`` and
    ``berreman._freeze`` already apply.  Walks tuples / lists / dicts."""
    if isinstance(obj, np.ndarray):
        if obj.flags.owndata or obj.base is None:
            obj.flags.writeable = False
        else:                          # a view: freeze what it looks through
            try:
                obj.flags.writeable = False
            except ValueError:         # pragma: no cover - non-freezable view
                pass
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            _freeze_cached(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _freeze_cached(v)
    return obj


def _mass_flux_cut(flux, W2, SVt, SVb, n, xp=np):
    """Propagating-mode cut for the MASS-WEIGHTED modal z-flux -- UNIT-SAFE.

    W7 B2 (2026-07-26).  ``flux = Im(E^T S0 conj(H))`` is contracted through
    the nodal mass ``S0``, whose entries carry the element JACOBIAN, so the
    whole flux spectrum scales LINEARLY with the ABSOLUTE period.  The
    historical floor was ``1e-9 * max(max|flux|, 1.0)`` -- and that ``1.0``
    has length units, pinning the cut at an ABSOLUTE ``1e-9``.  Harmless in
    micrometres; in METRES the entire spectrum sinks below it and EVERY
    propagating mode is reclassified evanescent, dropping the selector back to
    the legacy ``Im(q) < 0`` sign test that the v5.14 robustness audit
    replaced precisely because it spawns dense spurious resonances.

    Measured (degree 7, 28 DOF, in-plane tensor layer): ``max|flux| = 9.07e-2``
    at period 0.8 (um scale) but ``9.07e-10`` for the SAME 8 nm structure
    expressed in metres -- against a fixed ``1e-9``, 6 of 6 modes flip.
    Through ``pmm_jones_1d`` (8 nm period, 5.5 nm wavelength, 3.5 nm depth,
    degree 7) that is

        nanometres  -> sum(R) = 0.029337  sum(T) = 1.970666  R+T = 2.000003
        micrometres -> sum(R) = 0.029337  sum(T) = 1.970666  R+T = 2.000003
        metres      -> sum(R) = 0.036208  sum(T) = 1.991085  R+T = 2.027293

    -- a 1.4% energy-conservation violation produced by the UNIT CHOICE alone
    (max per-order drift 3.4e-2).  The metre break period grows with degree
    (5.2 nm at degree 5, 24.9 nm at degree 24, since more DOF means a smaller
    unit-norm eigenvector flux), and even far above it the margin is degraded:
    at an 800 nm period, degree 16, metres, ``max|flux| = 4.8e-8`` against
    ``1e-9`` is a 48x margin instead of the intended 1e9x.

    The floor is now the ROUND-OFF BOUND of the two contractions,
    ``max|W2| (max|SVt| + max|SVb|) n eps``, which carries the same Jacobian
    as ``flux`` and is therefore unit-invariant.  In micrometres it sits many
    decades below the ``1e-9 max|flux|`` term, so the cut is unchanged there.
    (``_build_generator_metric``'s selector is NOT routed here: its flux is a
    plain nodal sum with no ``S0``, hence already dimensionless.)"""
    fmax = xp.max(xp.abs(flux))
    fnoise = (xp.max(xp.abs(W2))
              * (xp.max(xp.abs(SVt)) + xp.max(xp.abs(SVb)))
              * (int(n) * float(np.finfo(np.float64).eps)))
    return xp.abs(flux) > xp.maximum(1e-9 * fmax, fnoise)


def _sem_modes(mats, k0, polarization, kx0=0.0, robust=False):
    """Periodic generalized eigenproblem on the nodal basis.

    Returns ``(Acoef, lam, q, invop)``: ``Acoef[:, n]`` = nodal values of mode
    ``n``'s field profile (``E_y`` TE / ``H_y`` TM); ``q = gamma/k0``;
    ``lam = -i q`` (forward-decaying propagator); ``invop`` = nodal multiply-by-
    ``1/eps`` operator (TM only).

    ``kx0`` (= incident transverse wavenumber, ``Re(n_sup) sin(angle) k0``) adds
    the Bloch shift of the pseudo-periodic envelope: the x-derivative becomes
    ``d/dx + i kx0``, so the stiffness picks up the ANTISYMMETRIZED convection
    ``-i kx0 (C - C^T)`` (``C = INT phi phi'``; for TM the 1/eps-weighted
    ``Cinv``, which is NOT antisymmetric across the wall -> the (Cinv - Cinv^T)
    form is required, not ``2 Cinv``) and the ``kx0^2`` mass.  At ``kx0 == 0``
    the shift vanishes.

    ``robust`` is ACCEPTED AND IGNORED (audit M10 2026-07-25 corrected this
    docstring, which used to describe it as selecting the branch): the
    NOISE-ROBUST forward selector has been UNCONDITIONAL since v5.14
    (robustness audit P1 -- the legacy ``Im(q) >= 0`` test flipped near-real
    propagating modes on ~1e-15 QZ noise and produced dense spurious
    resonances), so ``robust=False`` does NOT restore the legacy branch.  The
    parameter is retained only so the existing call sites (binary passes
    ``False``, segmented ``True``) keep their signatures; the same is true of
    :func:`_sem_modes_tensor`.
    """
    k02 = k0 * k0
    if polarization == "te":
        Lop = mats["L"]
        if kx0:
            Cas = mats["C"] - mats["C"].T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * mats["S0"]
        A, B = mats["Peps"] - Lop / k02, mats["S0"]
        invop = None
    else:
        Lop = mats["Linv"]
        if kx0:
            Cas = mats["Cinv"] - mats["Cinv"].T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * mats["Pinv"]
        A, B = mats["S0"] - Lop / k02, mats["Pinv"]
        invop = _safe_solve(mats["S0"], mats["Pinv"])
    q2, Acoef = _fast_geig(A, B)
    q = np.sqrt(q2)
    # NOISE-ROBUST forward branch (unconditional since v5.14, robustness audit
    # P1): the operator is (near-)Hermitian for lossless media, so the QZ eig
    # leaks ~1e-15 imag noise; the legacy naive sign test
    # ``q = where(q.imag < 0, -q, q)`` -- kept until v5.14 for the
    # normal-incidence binary path's bit-identity -- flipped near-real
    # (propagating) modes on that noise, producing DENSE spurious resonances at
    # normal incidence (8 of 13 degrees in 12..24 returned sum(R)+sum(T) up to
    # 65.7 on a plain n=2/1 grating).  The robust branch flips only when
    # CLEARLY backward and restores tot = 1.0 at every probed degree, matching
    # the RCWA oracle per-order to ~6e-6; oblique/segmented paths always used
    # it.  (``robust`` is retained in the signature for call compatibility.)
    q = _forward_branch_flip(q)                   # shared selector (S1-8)
    lam = -1j * q
    return Acoef, lam, q, invop


def _scalar_uniform_geo_eig(mats, k0, kx0=0.0):
    """Eps-free GEOMETRIC eig for a uniform-isotropic half-space (audit S5-P1).

    For uniform eps both the TE ``(eps*S0 - Lop/k0^2) x = q^2 S0 x`` and the TM
    ``(S0 - Linv/k0^2) x = q^2 Pinv x`` problems fold (Linv = L/eps, Pinv =
    S0/eps, Cinv = C/eps) to the SAME eps-FREE pencil
    ``(Lop/k0^2) x = (eps - q^2) S0 x`` with ``Lop = L - i*kx0*(C - C^T) +
    kx0^2*S0`` (geometry + Bloch only).  So ONE eig serves both half-spaces
    (sup/sub share the mesh -> identical L/C/S0) and both polarizations, with
    ``q^2 = eps - mu``.  Returns ``(mu, X)``."""
    k02 = k0 * k0
    Lop = mats["L"]
    if kx0:
        Cas = mats["C"] - mats["C"].T
        Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * mats["S0"]
    # The generalized pencil (Lop, S0) is k0-INDEPENDENT at FIXED kx0 (D14):
    # k0 enters purely as the 1/k0^2 scale.  eig(Lop/k0^2, S0) has eigenvalues
    # eig(Lop, S0)/k0^2 with the SAME eigenvectors, so eig the pencil ONCE
    # (cached) and scale.  NB the cache key bakes in the ABSOLUTE
    # kx0 = n*sin(angle)*k0, which scales with k0 -- so a fixed-ANGLE wavelength
    # sweep re-eigs every point; cross-wavelength reuse applies only at NORMAL
    # incidence (kx0 = 0).  Matches the historical per-k0 eig to ~1e-14 (the
    # physically-equivalent gauge; see _uniform_geo_eig).
    S0 = mats["S0"]
    mu_geo, X = _cached_geo_eig(
        (b"scalar", Lop.shape, Lop.tobytes(), S0.tobytes()),
        lambda: _fast_geig(Lop, S0))
    return mu_geo / k02, X


def _sem_modes_uniform_scalar(mats, k0, polarization, eps, kx0=0.0, geo=None):
    """Uniform-isotropic half-space modes from the shared geometric eig
    (audit S5-P1) -- the same ``(Acoef, lam, q, invop)`` contract as
    :func:`_sem_modes` for a uniform-eps cell, with ``q^2 = eps - mu`` and the
    IDENTICAL forward-branch selector + TM ``invop``.  The eigenvector gauge
    may differ from a raw per-eps :func:`_sem_modes` eig, but the downstream
    interface S-matrix is gauge-agnostic."""
    if geo is None:
        geo = _scalar_uniform_geo_eig(mats, k0, kx0)
    mu, X = geo
    q = np.sqrt(_C(eps) - mu)
    q = _forward_branch_flip(q)                   # shared selector (S1-8)
    lam = -1j * q
    invop = (None if polarization == "te"
             else _safe_solve(mats["S0"], mats["Pinv"]))
    return X, lam, q, invop


@functools.lru_cache(maxsize=64)
def _sem_projection_quad(degree, ref_nodes_key):
    """Degree-only Gauss quadrature + reference Lagrange values for
    :func:`_sem_fourier_projection` (audit P2).  ``leggauss`` (a Newton
    iteration), the barycentric weights and the Lagrange values at the
    quadrature nodes depend only on ``degree`` (the GLL reference nodes), NOT on
    the geometry -- cache them so every solve and every stabilize-scan step
    reuses them.  Keyed on ``ref_nodes_key`` (a tuple) as well so a different
    reference node set can never return stale values.  Returns read-only
    ``(xg, wg, Lv)``; callers must not mutate them (they don't)."""
    from numpy.polynomial.legendre import leggauss
    ref_nodes = np.asarray(ref_nodes_key, dtype=float)
    nq = max(2 * degree + 8, 24)
    xg, wg = leggauss(nq)
    wbary = np.ones(degree + 1)
    for j in range(degree + 1):
        for k in range(degree + 1):
            if k != j:
                wbary[j] /= (ref_nodes[j] - ref_nodes[k])
    Lv = np.zeros((len(xg), degree + 1))
    for r, x in enumerate(xg):
        diff = x - ref_nodes
        if np.any(np.abs(diff) < 1e-14):
            Lv[r, np.argmin(np.abs(diff))] = 1.0
        else:
            num = wbary / diff
            Lv[r, :] = num / num.sum()
    xg.flags.writeable = False
    wg.flags.writeable = False
    Lv.flags.writeable = False
    return xg, wg, Lv


def _sem_fourier_projection(orders, period, mats):
    """``T[m, i] = (1/period) INT phi_i(x) exp(-i m G x) dx`` for the global
    nodal Lagrange basis ``phi_i``; exact per element by oversampled Gauss
    quadrature (the integrand is oscillatory)."""
    l2g, elem_bnds, degree = mats["l2g"], mats["elem_bnds"], mats["degree"]
    ref_nodes, n_glob = mats["ref_nodes"], mats["n_glob"]
    G = 2.0 * np.pi / period
    xg, wg, Lv = _sem_projection_quad(
        int(degree), tuple(float(x) for x in ref_nodes))
    T = np.zeros((len(orders), n_glob), dtype=_C)
    for e in range(len(elem_bnds)):
        xl, xr, _eps = elem_bnds[e]
        J = 0.5 * (xr - xl)
        xphys = 0.5 * (xr + xl) + J * xg
        phase = np.exp(-1j * np.outer(orders * G, xphys))
        contrib = (phase * (wg * J / period)) @ Lv
        idx = l2g[e]
        for a in range(degree + 1):
            T[:, idx[a]] += contrib[:, a]
    return T



# ===========================================================================
# Self-contained scattering-matrix algebra (algebra-identical to rcwa's, kept
# local so this module stays in ONE convention with no cross-coupling)
# ===========================================================================

def _interface_smatrix(Wa, Va, Wb, Vb):
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    apb, amb = a + b, a - b
    iapb = np.linalg.inv(apb)
    return (-iapb @ amb, 2.0 * iapb,
            0.5 * (apb - amb @ iapb @ amb), amb @ iapb)



def _propagation_smatrix(lam, k0_L):
    n = lam.shape[0]
    X = np.diag(np.exp(-lam * k0_L))
    Z = np.zeros((n, n), dtype=_C)
    return (Z, X, X, Z)



def _kz_forward(eps, kx):
    """``kz/k0`` on the forward branch for ``exp(-i w t)``: ``Im(kz) >= 0`` so
    the forward wave ``exp(+i kz z)`` decays."""
    val = np.sqrt(np.asarray(eps - kx ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)



def _assemble_jones_farfield(Hsup, Hsub, S11, S21, orders, kx,
                             kz_sup, kz_sub, kz_inc, kx0n, N,
                             return_modal=False):
    """Rayleigh far-field bookkeeping for a 2x2 Jones grating solve.

    Given the Rayleigh-projection operators ``Hsup``/``Hsub`` (each a
    ``(2N, modes)`` operator stacking the x-orders over the y-orders), the solved
    interface S-matrix blocks ``S11``/``S21``, and the per-order ``kx``/``kz`` in
    the two homogeneous half-spaces, drive the cell with a unit order-0 ``Ex``
    (col 0) then ``Ey`` (col 1) and return ``(R_eff, T_eff, jones)``:

    * ``R_eff``/``T_eff`` -- ``(2, N)`` real diffraction efficiencies, one row per
      incident polarization, flux-normalized.  The longitudinal ``Ez`` from
      ``div D = 0`` in the isotropic half-space (``rz = -kx rx / kz``) carries the
      wall-normal (Ex / TM-like) z-flux -- the term that makes the TM channel
      conserve energy.  The col-0 (Ex, p-pol) incident wave itself carries
      ``Ez_inc = -kx0 Ex / kz_inc``, so its incident z-flux is
      ``kz_inc (1 + (kx0/kz_inc)^2)``; col-1 (Ey, s-pol) has ``Ez_inc = 0``.  At
      ``kx0 = 0`` both reduce to ``kz_inc`` (byte-identical to a normal solve).
    * ``jones`` -- the ``(2, 2)`` order-0 reflection Jones in the PUBLIC
      ``exp(-i w t)`` convention (no conjugation here; covariant callers conj the
      returned matrix to bridge their internal ``exp(+i w t)`` gauge).

    Extracted verbatim from the five identical NumPy Jones far-field loops
    (vertical core, convection slant, PMMStack general + covariant cascades, and
    the covariant single-layer core) so the flux bookkeeping lives in one place.
    ``kx0n = kx0 / k0`` is the (dimensionless) Bloch shift; callers compute
    ``kz_sup``/``kz_sub``/``kz_inc`` however their gauge requires and pass them in.

    ``return_modal=True`` (AUDIT_DYNAMETA_CONSUMER_API_GAPS B) additionally
    returns the per-order complex tangential amplitude dict (``rx``/``ry``/
    ``tx``/``ty`` each ``(2, N)``, rows keyed to incident lab ``E_x``/``E_y``,
    in the CALLER's gauge -- public for the classical ``PMMStack`` caller) --
    the data this helper already computes and squares into the efficiencies.
    """
    safe_r = np.where(np.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = np.where(np.abs(kz_sub) < 1e-12, 1.0, kz_sub)
    # TWO-SIDED + non-finite-aware (audit M3 2026-07-25): the former
    # ``abs(kz_inc) < 1e-9`` accepted a NEGATIVE kz_inc, which is exactly what a
    # GAIN superstrate produces (``_kz_forward`` takes its Re < 0 root), and
    # every efficiency is then silently NEGATED (measured tot = [-0.95, -0.82]
    # through the classical PMMStack cascade).  ``not (kz_inc > 1e-9)`` covers
    # grazing, negative AND NaN in one comparison, for all five callers.
    if not (kz_inc > 1e-9):
        raise ValueError(
            f"pmm: non-propagating incidence (kz_inc = {kz_inc:.6g}; needs "
            "> 0) -- either grazing/evanescent (kz_inc ~ 0: the incident wave "
            "carries ~no z-flux, so the R/T flux normalization is "
            "ill-defined), or a GAIN / non-propagating incidence medium whose "
            "forward root flips kz_inc negative and would negate every "
            "efficiency.  Reduce the incidence angle below grazing and use a "
            "lossless or lossy (Im(n_superstrate) >= 0) propagating "
            "superstrate.")
    m0 = np.where(orders == 0)[0][0]
    jones = np.zeros((2, 2), dtype=_C)
    R_eff = np.zeros((2, N))
    T_eff = np.zeros((2, N))
    amp = {k: np.zeros((2, N), dtype=_C) for k in ("rx", "ry", "tx", "ty")}
    for col in range(2):                        # 0 = incident Ex, 1 = incident Ey
        rhs = np.zeros(2 * N, dtype=_C)
        rhs[(col * N) + m0] = 1.0               # order-0 unit Ex (col 0) / Ey (col 1)
        cinc, *_ = np.linalg.lstsq(Hsup, rhs, rcond=None)
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:N], r_ord[N:]
        tx, ty = t_ord[:N], t_ord[N:]
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        flux_inc = kz_inc * (1.0 + (kx0n / kz_inc) ** 2) if col == 0 else kz_inc
        Re = np.real(kz_sup) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                + np.abs(rz) ** 2) / flux_inc
        Te = np.real(kz_sub) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                + np.abs(tz) ** 2) / flux_inc
        # Propagating-order mask: STRICTLY the cut-off ``Re(kz) > 0`` -- the ONE
        # threshold the whole family uses (rcwa ``_project_efficiency`` and the
        # 9 PMM 2-D / conical / JAX far-field sites), and the one this
        # function's own docstring documents.  Audit M7 2026-07-25: these five
        # 1-D PMM sites carried a ``> 1e-12`` floor instead, which differed only
        # for an order within 1e-12 of cut-off but made "below cut-off"
        # engine-dependent; measured zero effect on every pin.
        R_eff[col] = np.where(np.real(kz_sup) > 0.0, np.real(Re), 0.0)
        T_eff[col] = np.where(np.real(kz_sub) > 0.0, np.real(Te), 0.0)
        jones[0, col] = rx[m0]                  # PUBLIC convention -> no conjugation
        jones[1, col] = ry[m0]
        amp["rx"][col], amp["ry"][col] = rx, ry
        amp["tx"][col], amp["ty"][col] = tx, ty
    if return_modal:
        modal = dict(orders=np.asarray(orders).copy(), p0=int(m0),
                     kx=np.asarray(kx).copy(), ky=np.zeros(N),
                     kz_ref=np.asarray(kz_sup).copy(),
                     kz_trn=np.asarray(kz_sub).copy(),
                     kz_inc=float(kz_inc), kx0=float(kx0n), ky0=0.0, **amp)
        return R_eff, T_eff, jones, modal
    return R_eff, T_eff, jones


class PerOrderAmplitudesMixin:
    """Public per-order modal accessors over a ``self._modal`` slot
    (AUDIT_DYNAMETA_CONSUMER_API_GAPS B) -- the
    :meth:`~lumenairy.elements.rcwa.RCWAResult.per_order_amplitudes` contract
    mirrored for PMM-family builder classes.

    A solve path that closes through a Rayleigh far field stores the dict
    ``self._modal`` (keys ``orders``/``p0``/``rx ry tx ty`` each ``(2, N)``
    PUBLIC ``exp(-iwt)`` rows keyed to incident lab pol/``kx ky kz_ref
    kz_trn`` normalized by ``k0``/``kz_inc kx0 ky0``/``wavelength``); paths
    that keep no amplitudes (JAX twins, gauge-incompatible cascades) leave it
    ``None`` and the accessors raise.  Invalidate ``self._modal`` wherever
    the audit-P1-04 ``_internal`` contract invalidates."""

    _modal = None

    def per_order_amplitudes(self, port="reflection"):
        """Per-order complex tangential field amplitudes (PUBLIC
        ``exp(-iwt)`` convention) and transverse k-vectors of the LAST
        ``solve`` -- ``Ex``/``Ey`` each ``(2, N)`` (row 0 = response to
        incident lab ``E_x``, row 1 to ``E_y``), ``kx``/``ky``/``kz``
        normalized by ``k0``, the ``orders`` array (1-D ``(m,)`` or 2-D
        ``(N, 2)`` per the class's orders contract), and the
        ``wavelength``.  Raw TANGENTIAL amplitudes: an order's efficiency
        needs the flux weight ``Re(kz_m/kz_inc)``, the longitudinal
        ``Ez = -(kx Ex + ky Ey)/kz``, and the incident ``|E|^2`` (the recipe
        documented on ``RCWAResult.per_order_amplitudes``)."""
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"{type(self).__name__}.per_order_amplitudes: port must be "
                f"'reflection' or 'transmission', got {port!r}.")
        m = self._modal
        if m is None:
            raise ValueError(
                f"{type(self).__name__}.per_order_amplitudes: no per-order "
                f"amplitudes retained -- run a NumPy solve() first (any "
                f"add_layer / set_source / re-solve supersedes them; JAX "
                f"solves do not retain amplitudes).")
        ex, ey = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        kz = m["kz_ref"] if port == "reflection" else m["kz_trn"]
        return dict(orders=np.asarray(m["orders"]).copy(),
                    Ex=m[ex].copy(), Ey=m[ey].copy(),
                    kx=np.asarray(m["kx"]).copy(),
                    ky=np.asarray(m["ky"]).copy(),
                    kz=np.asarray(kz).copy(), wavelength=m["wavelength"],
                    # incidence terms for the jones_field_from_orders bridge
                    # (the transmission-port kz is the substrate kz) -- S5-5.
                    kz_inc=m["kz_inc"], kx0=m["kx0"], ky0=m["ky0"])

    def jones_transmission(self):
        """The ``(2, 2)`` zeroth-order TRANSMISSION Jones of the last
        ``solve`` (columns = incident lab ``E_x``/``E_y``, rows =
        ``[E_x; E_y]``, PUBLIC ``exp(-iwt)`` convention) -- the
        phase-bearing modulator observable.  Same availability as
        :meth:`per_order_amplitudes`."""
        m = self._modal
        if m is None:
            raise ValueError(
                f"{type(self).__name__}.jones_transmission: no per-order "
                f"amplitudes retained -- run a NumPy solve() first (JAX "
                f"solves do not retain amplitudes).")
        p0 = int(m["p0"])
        return np.stack([m["tx"][:, p0], m["ty"][:, p0]], axis=0)


def _scalar_farfield_RT(r_ord, t_ord, kx, kx0, k0, eps_sup, eps_sub,
                        polarization, label="pmm"):
    """TE/TM diffraction efficiencies from the order-resolved reflection /
    transmission amplitudes of a scalar PMM solve.

    The TE channel normalizes by ``kz`` (the standard plane-wave z-flux); the TM
    channel normalizes by the wall-normal flux ``kz/eps`` (the inverse-rule
    channel; the TM amplitudes are ``Hy``, and ``Re(kz/eps)|Hy|^2`` is the EXACT
    time-averaged ``Sz`` even in an absorbing medium, since
    ``Re(kz/eps) == Re(kz)(|kz|^2+kx^2)/|eps|^2``).  Orders below cut-off
    (``Re(kz) <= 0``) carry zero efficiency.  Shared verbatim by the vertical
    scalar core and the scalar slant solver.

    A GAIN superstrate (public ``Im(n_sup) < 0``) or a non-propagating
    (evanescent/metallic) incidence medium raises -- ``_kz_forward`` would flip
    ``kz_inc`` negative and silently negate every efficiency (the rcwa audit-P1
    guard, mirrored; ``eps_sup`` is PUBLIC here -> conj to internal)."""
    _require_propagating_incidence(label, np.conj(_C(eps_sup)),
                                   (kx0 / k0) ** 2)
    kz_sup = _kz_forward(eps_sup, kx)
    kz_sub = _kz_forward(eps_sub, kx)
    kz0 = complex(_kz_forward(eps_sup, np.array([kx0 / k0]))[0])
    kz_inc = float(np.real(kz0))
    # TWO-SIDED + non-finite-aware (audit M3 2026-07-25) -- see the identical
    # guard in _assemble_jones_farfield: a negative kz_inc (gain superstrate)
    # silently negates every efficiency, and NaN slips past a one-sided test.
    if not (kz_inc > 1e-9):
        raise ValueError(
            f"pmm: non-propagating incidence (kz_inc = {kz_inc:.6g}; needs "
            "> 0) -- grazing/evanescent, or a GAIN incidence medium whose "
            "forward root flips kz_inc negative and would negate every "
            "efficiency.  Reduce the incidence angle below grazing and use a "
            "lossless or lossy (Im(n_superstrate) >= 0) propagating "
            "superstrate.")
    if polarization == "te":
        R = np.real(kz_sup / kz_inc) * np.abs(r_ord) ** 2
        T = np.real(kz_sub / kz_inc) * np.abs(t_ord) ** 2
    else:
        flux_inc = np.real(kz_inc / eps_sup)
        if float(np.imag(_C(eps_sup))) != 0.0:
            # W7 F-B (2026-07-26): for an ABSORBING SUPERSTRATE the historical
            # ``Re(kz_inc/eps_sup)`` mixed gauges -- a REAL kz_inc (already
            # ``Re`` of the complex order-0 root) divided into a COMPLEX
            # eps_sup, which is the flux of no wave, while the NUMERATORS use
            # the full complex kz.  It broke the hardest symmetry there is: at
            # NORMAL incidence on an ISOTROPIC slab, TE and TM must be
            # identical, and they were not (measured T drift 1.4e-4 at
            # Im(n_sup)=0.01, 3.4e-3 at 0.05, 5.8e-2 at 0.2), so this ONE
            # recipe disagreed with every other far field in the family
            # (rcwa ``_project_efficiency``, ``_assemble_jones_farfield``, the
            # 2-D/conical sites) on the SAME physical problem.
            #
            # The family normalizes an E-amplitude flux by
            # ``kz_inc * einc_sq``; in the TM channel's Hy gauge that incident
            # flux is ``(kz_inc^2+kx0^2)/kz_inc * |kz0/eps_sup|^2`` (the
            # ``|Ex_inc|^2 = |kz0/eps_sup|^2`` conversion for unit Hy).  For a
            # REAL eps_sup this equals ``kz_inc/eps_sup`` identically, so the
            # branch is skipped and every lossless solve is BYTE-UNCHANGED.
            # Post-fix: rcwa parity 1.3e-14, TE == TM at normal 2.7e-15.
            flux_inc = (((kz_inc ** 2 + (kx0 / k0) ** 2) / kz_inc)
                        * abs(kz0 / _C(eps_sup)) ** 2)
        R = np.real(kz_sup / eps_sup) * np.abs(r_ord) ** 2 / flux_inc
        T = np.real(kz_sub / eps_sub) * np.abs(t_ord) ** 2 / flux_inc
    # ``Re(kz) > 0`` -- the family-wide cut-off mask (audit M7; see
    # _assemble_jones_farfield for the measurement).
    R = np.where(np.real(kz_sup) > 0.0, np.real(R), 0.0)
    T = np.where(np.real(kz_sub) > 0.0, np.real(T), 0.0)
    return R, T



# ===========================================================================
# Core single-polarization PMM solve
# ===========================================================================

def _n_propagating_orders(period, wl, n_max):
    """Highest propagating Rayleigh order |m| with |kx_m| < n_max*k0."""
    return int(np.floor(float(np.real(n_max)) * period / wl + 1e-9))



def _wood_safe_wl_1d(wl, angle, n_sup, period, eps_values, far_field_orders):
    """1-D Wood-anomaly wavelength nudge (v5.14 robustness audit): a
    wavelength sitting EXACTLY on (or within ~1e-9 of) a Rayleigh-order cutoff
    in any constituent medium puts a grazing order (``kz ~ 0``) in the flux
    normalization and silently violates energy conservation (measured
    ``tot = 1.00025`` at ``wl = P*(1 - 1e-9)`` with no warning).  The 2-D
    paths already nudge; this is the 1-D counterpart (identity away from exact
    grazing, so ordinary solves are byte-unchanged)."""
    from ..rcwa._core import _grazing_safe_wavelength
    kx0n = float(np.real(_C(n_sup))) * np.sin(float(angle))  # dimensionless
    m = np.arange(-int(far_field_orders), int(far_field_orders) + 1)
    return _grazing_safe_wavelength(float(wl), kx0n, 0.0, m,
                                    np.zeros_like(m), period, period,
                                    [complex(e) for e in eps_values])


def _pmm_solve(period, n_ridge, n_groove, n_sub, n_sup, depth, duty, wl,
               degree, polarization, n_ridge_el, n_groove_el, grade,
               far_field_orders, angle=0.0):
    """Binary (ridge/groove) single-degree scalar PMM solve -- a thin wrapper
    that builds the two-region operators and defers to :func:`_pmm_solve_core`
    (shared with the multi-region :func:`_pmm_solve_segments`)."""
    eps_ridge, eps_groove = n_ridge ** 2, n_groove ** 2
    eps_sup, eps_sub = n_sup ** 2, n_sub ** 2
    wl = _wood_safe_wl_1d(wl, angle, n_sup, period,
                          [eps_sup, eps_sub, eps_ridge, eps_groove],
                          far_field_orders)
    k0 = 2.0 * np.pi / wl
    d_wall = duty * period
    # NB: the 1-D PMM ``kx0`` is DIMENSIONAL (rad/m, the ``* k0`` factor) -- it is
    # the physical Bloch wavenumber added to d/dx.  This differs from the RCWA 2-D
    # convention where ``kx0 = n sin(theta) cos(phi)`` is DIMENSIONLESS (k0-
    # normalised, like Kx/Ky); do not cross-wire the two.
    kx0 = float(np.real(n_sup)) * np.sin(float(angle)) * k0

    mats = _build_sem(period, d_wall, eps_ridge, eps_groove, degree,
                      n_ridge_el, n_groove_el, grade)
    mats_sup = _build_sem(period, d_wall, eps_sup, eps_sup, degree,
                          n_ridge_el, n_groove_el, grade)
    mats_sub = _build_sem(period, d_wall, eps_sub, eps_sub, degree,
                          n_ridge_el, n_groove_el, grade)
    n_max = max(np.real(n_sup), np.real(n_sub), np.real(n_ridge),
                np.real(n_groove))
    return _pmm_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max,
                           period, depth, wl, degree, polarization,
                           far_field_orders, kx0)



def _pmm_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max, period,
                    depth, wl, degree, polarization, far_field_orders, kx0,
                    label="pmm_efficiency_1d", robust=False):
    """Single-degree scalar PMM solve from PRE-BUILT spectral-element operators
    (the layer ``mats`` and the matching homogeneous half-space ``mats_sup`` /
    ``mats_sub``, all sharing the same node layout).  Shared by the binary
    :func:`_pmm_solve` and the multi-region :func:`_pmm_solve_segments`
    (``robust`` is accepted and IGNORED -- the noise-robust forward branch is
    unconditional since v5.14; see :func:`_sem_modes`)."""
    k0 = 2.0 * np.pi / wl
    n_glob = mats["n_glob"]
    # Rayleigh order set for the (forward-only) far-field projection: cover the
    # propagating orders with an evanescent buffer, kept WELL BELOW n_glob (a
    # projection order count approaching n_glob aliases the nodal->Fourier map).
    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: degree={degree} too low to resolve the "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
            f"degree or elements_per_region.")
    orders = np.arange(-half, half + 1)
    G = 2.0 * np.pi / period
    kx = (kx0 + orders * G) / k0                     # oblique: kx_m = (kx0+mG)/k0
    Tp = _sem_fourier_projection(orders, period, mats)

    Acoef, lam_l, q_l, invop = _sem_modes(mats, k0, polarization, kx0, robust)
    Wl = Acoef
    Vl = (Acoef if polarization == "te" else invop @ Acoef) @ np.diag(q_l)

    # audit S5-P1: the two UNIFORM sup/sub half-spaces share ONE eps-free
    # geometric eig (they share the mesh -> identical L/C/S0) instead of two
    # full _sem_modes eigs -- the half-spaces were 51-64% of the 1-D eig time.
    _geo = _scalar_uniform_geo_eig(mats_sup, k0, kx0)
    Wsup, _ls, q_sup, invsup = _sem_modes_uniform_scalar(
        mats_sup, k0, polarization, eps_sup, kx0, geo=_geo)
    Wsub, _lb, q_sub, invsub = _sem_modes_uniform_scalar(
        mats_sub, k0, polarization, eps_sub, kx0, geo=_geo)
    if polarization == "te":
        Vsup, Vsub = Wsup @ np.diag(q_sup), Wsub @ np.diag(q_sub)
    else:
        Vsup = (invsup @ Wsup) @ np.diag(q_sup)
        Vsub = (invsub @ Wsub) @ np.diag(q_sub)
    # (region half-spaces do not propagate, so no region lam is needed.)

    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _propagation_star(S, lam_l, k0 * depth)
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # incident order-0 plane wave -> superstrate mode amplitudes (forward
    # projection of the region modes onto Rayleigh orders, then least squares).
    Hsup = Tp @ Wsup
    Hsub = Tp @ Wsub
    delta0 = (orders == 0).astype(_C)
    cinc, *_ = np.linalg.lstsq(Hsup, delta0, rcond=None)
    r_ord = Hsup @ (S11 @ cinc)
    t_ord = Hsub @ (S21 @ cinc)

    R, T = _scalar_farfield_RT(r_ord, t_ord, kx, kx0, k0, eps_sup, eps_sub,
                               polarization, label=label)
    return orders, R, T, n_glob



# ===========================================================================
# Anisotropic (in-plane tensor) PMM -- the full 2x2 Jones reflection
# ===========================================================================
#
# The spectral-element counterpart of the FMM tensor path (rcwa_jones_1d).  The
# modal field is a 2-vector [E_x; E_y] per node; the in-plane tensor
#   eps = [[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]]
# couples E_x <-> E_y through exy / eyx.  Mirroring the FMM eigenmode block at
# normal incidence (Ky = 0), the coupled SECOND-ORDER modal eigenproblem is
#
#     M @ [Ex; Ey] = q^2 [Ex; Ey],   q = gamma / k0 = n_eff,   M = -P@Q,
#     M = [[ G Cxx ,  G Cxy ],
#          [ Cyx   ,  Cyy - Kx^2 ]],   G = I - Kx Ezzi Kx,
#
# with the Li-1996 in-plane factorization (the wall normal is x):
#     Cxx = [[1/exx]]^-1,   Cxy = Cxx [[exy/exx]],   Cyx = [[eyx/exx]] Cxx,
#     Cyy = [[eyy - eyx exy/exx]] + [[eyx/exx]] Cxx [[exy/exx]].
#
# SPECTRAL-ELEMENT realization (the inverse rule is AUTOMATIC and EXACT because
# every eps jump lands on an element boundary):
#   * a Fourier convolution [[f]] becomes the nodal multiply operator
#     hat(f) = S0^-1 (INT phi f phi)  (piecewise-constant f -> essentially
#     diagonal, the shared wall node carrying the exact C0 average);
#   * the wall-normal inverse rule [[1/exx]]^-1 becomes inv(hat(1/exx));
#   * the Kx-derivative terms become spectral-element STIFFNESS operators -- the
#     weak form moves one derivative onto the test function, integrating the wall
#     jump exactly:  Kx (w) Kx -> (1/k0^2) S0^-1 (INT phi' w phi').  The
#     Ez-elimination G uses the 1/ezz-weighted stiffness (the Ez wall-tangential
#     inverse-rule = inv([[ezz]])); the plain Kx^2 uses the unit-weighted one.
#
# The modal magnetic partner V = Q @ W @ diag(1/lam) (the Ky=0 Q block) feeds the
# SAME interface S-matrix as the scalar path.  PUBLIC exp(-i w t) convention
# end-to-end (no eps conjugation, no Jones conjugation) -- the scalar PMM is
# self-contained in the public convention, and conjugating in+out (as the FMM
# oracle does internally) would double-flip to conj(J).  Energy validated.

# anisotropic-Jones stabilize parameters: both incident-polarization totals must
# be passive (each <= 1 + tol) to reject the isolated-degree resonances.
_JONES_PASSIVE_TOL = 2.0e-3


def _build_sem_tensor(period, d_wall, t_ridge, t_groove, degree,
                      n_ridge_el, n_groove_el, grade):
    """Assemble the periodic C0 spectral-element operators for an ANISOTROPIC
    layer (per-coefficient masses + weighted stiffnesses).

    ``t_ridge`` / ``t_groove`` are dicts ``dict(exx, exy, eyx, eyy, ezz)`` of the
    (already convention-correct) tensor components, constant within each region.
    Returns a dict with the nodal mass operators (``INT phi c phi`` for each
    coefficient ``c``) and stiffness operators (``INT phi' c phi'``), plus the
    local->global map / element table (shared with the scalar projection).
    """
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    rb = _graded_boundaries(0.0, d_wall, n_ridge_el, grade)
    gb = _graded_boundaries(d_wall, period, n_groove_el, grade)
    elem_bnds = (list(zip(rb[:-1], rb[1:], [t_ridge] * n_ridge_el))
                 + list(zip(gb[:-1], gb[1:], [t_groove] * n_groove_el)))
    n_el = len(elem_bnds)

    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n_glob = last

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)

    # mass operators keyed by coefficient; stiffness operators keyed by weight
    mass = {k: _z() for k in ("one", "eyy", "exy_xx", "eyx_xx", "schur", "ezz",
                              "inv_xx", "inv_ezz")}
    stiff = {k: _z() for k in ("one", "inv_ezz")}
    conv = {k: _z() for k in ("one", "inv_ezz")}    # convection (oblique only)
    for e in range(n_el):
        xl, xr, t = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Kloc = (Dphys.T * wel) @ Dphys
        Cloc = Mloc @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        iez = 1.0 / ezz
        mass["one"][ix] += Mloc
        mass["eyy"][ix] += eyy * Mloc
        mass["exy_xx"][ix] += (exy / exx) * Mloc
        mass["eyx_xx"][ix] += (eyx / exx) * Mloc
        mass["schur"][ix] += (eyy - eyx * exy / exx) * Mloc
        mass["ezz"][ix] += ezz * Mloc
        mass["inv_xx"][ix] += (1.0 / exx) * Mloc
        mass["inv_ezz"][ix] += iez * Mloc
        stiff["one"][ix] += Kloc
        stiff["inv_ezz"][ix] += iez * Kloc
        conv["one"][ix] += Cloc
        conv["inv_ezz"][ix] += iez * Cloc
    return dict(mass=mass, stiff=stiff, conv=conv, S0=mass["one"],
                n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds, degree=degree,
                ref_nodes=ref_nodes)



def _sem_modes_tensor(mats, k0, kx0=0.0, robust=False, ky0=0.0):
    """Coupled ``(E_x, E_y)`` anisotropic SE modal eigenproblem.

    Returns ``(W2, V2, lam, q)``: ``W2[:, n]`` = the 2-vector field of mode
    ``n`` (top ``n_glob`` rows ``E_x``, bottom ``E_y``); ``V2`` the matched
    magnetic partner ``Q W diag(1/lam)``; ``q = gamma/k0``;
    ``lam = -i q`` (forward-decaying propagator).

    ``kx0`` (the incident transverse wavenumber) Bloch-shifts every ``Kx``
    operator: ``Kx^2`` (unit weight) and ``Kx (1/ezz) Kx`` (Ez elimination) each
    become ``(Kx + kx0) w (Kx + kx0)``, i.e. the weak-form stiffness gains
    ``-i kx0 (conv_w - conv_w^T) + kx0^2 mass_w``.  At ``kx0 == 0`` this is
    bit-identical to the normal-incidence solve (shift vanishes, legacy
    ``Im(q) >= 0`` branch).  At ``kx0 != 0`` the coupled modes are genuinely
    non-Hermitian so the forward set is chosen by the z-POYNTING FLUX sign (the
    Im(q) split forms an inconsistent, non-passive basis there).

    ``ky0`` (DIMENSIONAL, rad/m -- same units as ``kx0``) generalizes the
    eigenproblem to CONICAL incidence
    (``AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12``): the structure is
    uniform along y, so ``Ky = (ky0/k0) I`` is a SCALAR in the nodal basis and
    the full dimension-agnostic tensor blocks (the validated
    ``rcwa._layer_eigenmodes_tensor`` P/Q form) assemble from the SAME weak
    operators::

        P = [[Ky Kx(1/ezz),   I - Kx(1/ezz)Kx],     Q = [[Cyx + Ky Kx,   Cyy - Kx^2],
             [Ky^2 (1/ezz) - I,  -Ky (1/ezz)Kx]]         [Ky^2 - Cxx,  -(Cxy + Ky Kx)]]

        M = -(P @ Q),   M v = q^2 v,   V = Q W diag(1/lam).

    The single-``Kx`` cross terms are elementwise-EXACT weak first-derivative
    forms (``conv`` operators; the integrand is polynomial within every
    element, so wall kinks cost nothing).  Every added term carries a ``ky0``
    factor, so at ``ky0 == 0`` this is BIT-IDENTICAL to the classical solve --
    the degenerate-limit reduction the conical path requires.  This is the
    no-projection-floor (pure nodal) conical layer build; the previous
    Fourier-projected route carried a resolution-independent ~3e-3 systematic
    error for patterned cells.
    """
    n = mats["n_glob"]
    k02 = k0 * k0
    S0 = mats["S0"]
    iS0 = _safe_inv(S0)
    mass, stiff, conv = mats["mass"], mats["stiff"], mats["conv"]

    # nodal pointwise operators (S0^-1 . Galerkin operator)
    Cinv_xx = iS0 @ mass["inv_xx"]          # multiply by 1/exx == [[1/exx]]
    Cxx = _safe_inv(Cinv_xx)                # [[1/exx]]^-1 (wall-normal inverse rule)
    EXY_XX = iS0 @ mass["exy_xx"]           # [[exy/exx]]
    EYX_XX = iS0 @ mass["eyx_xx"]           # [[eyx/exx]]
    SCHUR = iS0 @ mass["schur"]             # [[eyy - eyx exy/exx]]
    Cxy = Cxx @ EXY_XX
    Cyx = EYX_XX @ Cxx
    Cyy = SCHUR + EYX_XX @ Cxx @ EXY_XX

    # Kx-derivative operators (1/k0^2 . SE stiffness).  Ez-elimination uses the
    # 1/ezz-weighted stiffness (the Ez wall-tangential inverse rule); plain Kx^2
    # the unit-weighted one.  Bloch shift (kx0 != 0): (Kx+kx0) w (Kx+kx0).
    def _kxop(skey, ckey, mkey):
        op = stiff[skey]
        if kx0:
            Cw = conv[ckey]
            op = op - 1j * kx0 * (Cw - Cw.T) + (kx0 * kx0) * mass[mkey]
        return (1.0 / k02) * (iS0 @ op)
    KxEzziKx = _kxop("inv_ezz", "inv_ezz", "inv_ezz")
    Kx2 = _kxop("one", "one", "one")
    G = np.eye(n, dtype=_C) - KxEzziKx

    if ky0 == 0.0:
        Mbig = np.block([[G @ Cxx, G @ Cxy],
                         [Cyx,     Cyy - Kx2]])
        # modal magnetic partner: V = Q @ W @ diag(1/lam), (Ky=0) Q block
        Q = np.block([[Cyx, Cyy - Kx2], [-Cxx, -Cxy]])
    else:
        # ---- CONICAL (ky0 != 0) generalization: assemble the full P/Q of the
        # dimension-agnostic tensor form with Ky = (ky0/k0) I scalar and the
        # weak-form nodal Kx pieces (see docstring).  kyn is dimensionless.
        kyn = ky0 / k0
        I_n = np.eye(n, dtype=_C)
        EZI = iS0 @ mass["inv_ezz"]                     # multiply by 1/ezz
        # normalized weak first-derivative operators (elementwise-exact):
        #   Kx        = (-i/k0) d/dx + kxn            (Bloch-shifted)
        #   Kx(1/ezz) = d/dx o (1/ezz .)  (by parts: +conv_w^T) + kxn (1/ezz)
        #   (1/ezz)Kx = (1/ezz .) o d/dx  (direct conv_w)       + kxn (1/ezz)
        kxn = kx0 / k0
        Dx1 = iS0 @ conv["one"]
        Dxz = iS0 @ conv["inv_ezz"]
        DxzT = iS0 @ conv["inv_ezz"].T
        Kx1 = (-1j / k0) * Dx1 + kxn * I_n
        KxEZI = (1j / k0) * DxzT + kxn * EZI
        EZIKx = (-1j / k0) * Dxz + kxn * EZI
        P = np.block([
            [kyn * KxEZI,              G],
            [(kyn * kyn) * EZI - I_n,  -kyn * EZIKx],
        ])
        Q = np.block([
            [Cyx + kyn * Kx1,          Cyy - Kx2],
            [(kyn * kyn) * I_n - Cxx,  -(Cxy + kyn * Kx1)],
        ])
        Mbig = -(P @ Q)
    q2, W2 = np.linalg.eig(Mbig)
    q = np.sqrt(q2)

    # POYNTING-FLUX forward selector (unconditional since v5.14, robustness
    # audit P1 -- the legacy normal-incidence ``Im(q) >= 0`` branch flipped
    # degenerate propagating modes on ~1e-15 QZ noise, producing DENSE
    # spurious resonances: totals up to 344 at isolated degrees on a plain
    # diagonal-tensor grating).  V2 partner is [Hx; Hy] and the modal H
    # carries an extra -i, so Sz_n = Im( Ex.S0.conj(Hy) - Ey.S0.conj(Hx) )
    # (cross pairing, imag part: + forward, ~0 evanescent).  Flux ~ 1/q
    # flips sign with the branch; pick +z power (propagating) / +z decay.
    lam0 = -1j * q
    safe0 = np.where(np.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = Q @ W2 @ np.diag(1.0 / safe0)
    SVt = S0 @ np.conj(V0[:n])          # S0 conj(Hx)
    SVb = S0 @ np.conj(V0[n:])          # S0 conj(Hy)
    flux = np.imag(np.einsum("in,in->n", W2[:n], SVb)
                   - np.einsum("in,in->n", W2[n:], SVt))
    prop = _mass_flux_cut(flux, W2, SVt, SVb, n)      # W7 B2: unit-safe cut
    flip = np.where(prop, flux < 0.0, q.imag < 0.0)
    q = np.where(flip, -q, q)
    lam = -1j * q
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    V2 = Q @ W2 @ np.diag(1.0 / safe)
    return W2, V2, lam, q



# --- geometric-eig cache (wavelength-independent at fixed angle) --------------
# The eps-free geometric operator B = inv(S0) @ op depends only on the nodal
# GEOMETRY (mats) and kx0 (angle), NOT on k0 (wavelength): the k0 enters purely
# as the 1/k0^2 scale on B.  So a fixed-angle wavelength sweep re-eigs the SAME
# B every point.  Eig B ONCE (cached on its bytes) and scale the spectrum by
# 1/k0^2 -- this removes the half-space geometric eig (the audited 51-64% of 1-D
# eig time) from every sweep point after the first.  (NB the modes then match
# the historical per-k0 eig of B/k0^2 to ~1e-14 -- eig(cB) and eig(B) share
# eigenVECTORS to machine precision, exact eigenvalue scaling -- rather than
# bit-for-bit; a physically-equivalent gauge, as with the even-parity fold.)
_GEO_EIG_CACHE: 'OrderedDict[bytes, tuple]' = OrderedDict()
_GEO_EIG_CACHE_SIZE = 64
_GEO_EIG_CACHE_LOCK = threading.Lock()


def _clear_geo_eig_cache():
    """Registry hook: drop the geometric-eig cache."""
    with _GEO_EIG_CACHE_LOCK:
        _GEO_EIG_CACHE.clear()


def _cached_geo_eig(key, compute):
    """Memoize the k0-independent geometric eig ``compute()`` on ``key`` (a
    bytes fingerprint of the geometry+angle pencil).  Bounded LRU.

    The cached arrays are handed to callers BY IDENTITY, so they are marked
    READ-ONLY with the module's :func:`_readonly` guard (audit M9 2026-07-25 --
    the other cache sites already do this): an accidental in-place write on a
    returned eigenvector block would otherwise poison the cache for every
    later solve on the same geometry (measured: ``w[0,0] += 1`` changed the
    value the next two cache hits saw).  Callers only read / matmul these."""
    with _GEO_EIG_CACHE_LOCK:
        hit = _GEO_EIG_CACHE.get(key)
        if hit is not None:
            _GEO_EIG_CACHE.move_to_end(key)          # LRU: refresh recency
            return hit
    res = compute()
    res = tuple(_readonly(a) if isinstance(a, np.ndarray) else a for a in res)
    with _GEO_EIG_CACHE_LOCK:
        _GEO_EIG_CACHE[key] = res
        while len(_GEO_EIG_CACHE) > _GEO_EIG_CACHE_SIZE:
            _GEO_EIG_CACHE.popitem(last=False)
    return res


def _uniform_geo_eig(mats, k0, kx0=0.0):
    """Eps-free GEOMETRIC eigendecomposition shared by every uniform
    isotropic medium on one nodal grid (backlog A2 / PMM roadmap #2,
    2026-06-10).

    For a uniform isotropic ``eps`` the coupled operator of
    :func:`_sem_modes_tensor` collapses EXACTLY (no approximation)::

        Cxx = Cyy = eps * I,  Cxy = Cyx = 0,
        KxEzziKx = Kx2 / eps          (the 1/ezz weights are 1/eps * unit)
        => Mbig(eps) = eps * I_(2n) - blockdiag(Kx2, Kx2)

    so ONE eig of the n x n geometric operator ``Kx2`` serves every uniform
    medium: shared eigenvectors, spectrum ``q^2(eps) = eps - mu_geo``.  The
    2n problem block-diagonalizes into two identical n-blocks, so this is
    ~16x cheaper than two independent 2n eigs (the audited 51-64% of 1-D
    eig time spent on half-spaces).  Returns ``(mu, w)``.
    """
    k02 = k0 * k0
    iS0 = _safe_inv(mats["S0"])
    op = mats["stiff"]["one"]
    if kx0:
        Cw = mats["conv"]["one"]
        op = op - 1j * kx0 * (Cw - Cw.T) + (kx0 * kx0) * mats["mass"]["one"]
    # B is independent of k0 ONLY at fixed kx0 (D14): the LRU key bakes in the
    # ABSOLUTE kx0 = n*sin(theta)*k0 (via op above), which itself scales with
    # k0, so a FIXED-ANGLE wavelength sweep changes kx0 every point and the
    # cache re-eigs throughout -- cross-wavelength reuse exists only at NORMAL
    # incidence (kx0 = 0).  Always correct; the eig cannot be angle-normalized
    # without operator rescaling.  k0 then enters purely as the 1/k0^2 scale.
    B = iS0 @ op
    mu_geo, w = _cached_geo_eig(
        (b"tensor", B.shape, B.tobytes()), lambda: np.linalg.eig(B))
    return mu_geo / k02, w, B / k02


def _sem_modes_uniform(mats, k0, kx0, eps, geo=None):
    """Uniform-isotropic modes from the SHARED geometric eig (see
    :func:`_uniform_geo_eig`): same return contract as
    :func:`_sem_modes_tensor` -- ``(W2, V2, lam, q)`` with the IDENTICAL
    z-Poynting-flux forward selector, evaluated on the eps-shifted spectrum
    (the branch choice legitimately depends on eps: a mode propagating in
    the substrate may be evanescent in air).

    The eigenvector GAUGE differs from a raw 2n eig (clean per-block
    ``(w, 0)`` / ``(0, w)`` columns instead of arbitrary degenerate
    mixtures) -- physically equivalent; downstream interface solves are
    basis-agnostic.  Cross-path contracts follow docs/TOLERANCE_POLICY.md.
    """
    n = mats["n_glob"]
    eps = _C(eps)
    if geo is None:
        geo = _uniform_geo_eig(mats, k0, kx0)
    mu, w, _Kx2 = geo
    q2 = eps - mu
    q = np.sqrt(np.concatenate([q2, q2]))
    Z = np.zeros((n, n), dtype=_C)
    W2 = np.block([[w, Z], [Z, w]])
    # Q @ W2 for the uniform medium: Q = [[0, eps I - Kx2], [-eps I, 0]] and
    # Kx2 w = w diag(mu)  =>  (eps I - Kx2) w = w diag(q2)
    QW = np.block([[Z, w * q2[None, :]], [-eps * w, Z]])
    S0 = mats["S0"]
    # --- the _sem_modes_tensor forward selector, verbatim policy ----------
    lam0 = -1j * q
    safe0 = np.where(np.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = QW * (1.0 / safe0)[None, :]
    SVt = S0 @ np.conj(V0[:n])          # S0 conj(Hx)
    SVb = S0 @ np.conj(V0[n:])          # S0 conj(Hy)
    flux = np.imag(np.einsum("in,in->n", W2[:n], SVb)
                   - np.einsum("in,in->n", W2[n:], SVt))
    prop = _mass_flux_cut(flux, W2, SVt, SVb, n)      # W7 B2: unit-safe cut
    flip = np.where(prop, flux < 0.0, q.imag < 0.0)
    q = np.where(flip, -q, q)
    lam = -1j * q
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    V2 = QW * (1.0 / safe)[None, :]
    return W2, V2, lam, q


def _pmm_jones_solve(period, eps_ridge3, eps_groove3, n_sub, n_sup, depth,
                     duty, wl, degree, n_ridge_el, n_groove_el, grade,
                     far_field_orders, angle=0.0):
    """Single-degree coupled anisotropic PMM solve.

    Returns ``(orders, R(2,N), T(2,N), jones(2,2), n_glob)`` in the PUBLIC
    ``exp(-i w t)`` convention.  Row/column 0 is the incident ``E_x`` response,
    1 the incident ``E_y``; ``jones`` columns are the zeroth-order reflected
    ``[E_x; E_y]`` for incident ``E_x`` / ``E_y``.  ``angle != 0`` adds the
    ``kx0`` Bloch shift (modes) and the oblique far-field normalization.
    """
    er = np.asarray(eps_ridge3, dtype=_C)
    eg = np.asarray(eps_groove3, dtype=_C)

    def _t3(M):
        return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1],
                    ezz=M[2, 2])
    t_ridge, t_groove = _t3(er), _t3(eg)
    eps_sup, eps_sub = _C(n_sup) ** 2, _C(n_sub) ** 2
    wl = _wood_safe_wl_1d(wl, angle, n_sup, period,
                          [eps_sup, eps_sub, er[0, 0], er[1, 1], er[2, 2],
                           eg[0, 0], eg[1, 1], eg[2, 2]], far_field_orders)
    k0 = 2.0 * np.pi / wl
    d_wall = duty * period
    kx0 = float(np.real(_C(n_sup))) * np.sin(float(angle)) * k0

    mats = _build_sem_tensor(period, d_wall, t_ridge, t_groove, degree,
                             n_ridge_el, n_groove_el, grade)
    t_sup = dict(exx=eps_sup, exy=0.0, eyx=0.0, eyy=eps_sup, ezz=eps_sup)
    t_sub = dict(exx=eps_sub, exy=0.0, eyx=0.0, eyy=eps_sub, ezz=eps_sub)
    mats_sup = _build_sem_tensor(period, d_wall, t_sup, t_sup, degree,
                                 n_ridge_el, n_groove_el, grade)
    mats_sub = _build_sem_tensor(period, d_wall, t_sub, t_sub, degree,
                                 n_ridge_el, n_groove_el, grade)
    n_max = max(np.real(np.sqrt(eps_sup)), np.real(np.sqrt(eps_sub)),
                np.real(np.sqrt(er[0, 0])), np.real(np.sqrt(eg[0, 0])),
                np.real(np.sqrt(er[1, 1])), np.real(np.sqrt(eg[1, 1])))
    return _pmm_jones_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub,
                                 n_max, period, depth, wl, degree,
                                 far_field_orders, kx0)



def _pmm_jones_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max,
                          period, depth, wl, degree, far_field_orders, kx0,
                          label="pmm_jones_1d", robust=False):
    """Single-degree coupled anisotropic solve from PRE-BUILT operators (layer +
    matching homogeneous half-spaces); shared by the binary
    :func:`_pmm_jones_solve` and the multi-region
    :func:`_pmm_jones_solve_segments` (``robust`` forces the z-Poynting-flux
    forward selector even at normal incidence -- see :func:`_sem_modes_tensor`)."""
    k0 = 2.0 * np.pi / wl
    n_glob = mats["n_glob"]
    Wl, Vl, lam_l, _ql = _sem_modes_tensor(mats, k0, kx0, robust)
    # uniform half-spaces share ONE geometric eig (backlog A2): the n x n
    # spectrum is eps-shifted per medium -- ~16x cheaper than two 2n eigs
    _geo = _uniform_geo_eig(mats_sup, k0, kx0)
    Wsup, Vsup, _ls, _qs = _sem_modes_uniform(mats_sup, k0, kx0, eps_sup,
                                              _geo)
    Wsub, Vsub, _lb, _qb = _sem_modes_uniform(mats_sub, k0, kx0, eps_sub,
                                              _geo)

    # Rayleigh order set for the forward far-field projection (cover the
    # propagating orders, kept well below the nodal DOF -- see the scalar path).
    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: degree={degree} too low to resolve the "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
            f"degree or elements_per_region.")
    orders = np.arange(-half, half + 1)
    G = 2.0 * np.pi / period
    kx = (kx0 + orders * G) / k0
    N = len(orders)
    Tp = _sem_fourier_projection(orders, period, mats)

    # interface + propagation S-matrix (block size 2*n_glob; the field stacks
    # [Ex_nodal; Ey_nodal], so each mode matrix is already 2*n_glob tall).
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _propagation_star(S, lam_l, k0 * depth)
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    def _project(Wmodes):
        """Project each (Ex / Ey) half of the nodal modes onto the Rayleigh
        orders -> a (2N, modes) operator: x-orders stacked over y-orders."""
        return np.vstack([Tp @ Wmodes[:n_glob, :], Tp @ Wmodes[n_glob:, :]])

    Hsup = _project(Wsup)
    Hsub = _project(Wsub)

    # Gain / non-propagating incidence-medium guard (rcwa audit-P1 mirror;
    # PUBLIC-convention eps_sup here -> conj to the guard's internal gauge).
    _require_propagating_incidence(label, np.conj(_C(eps_sup)),
                                   (kx0 / k0) ** 2)
    kz_sup = _kz_forward(eps_sup, kx)
    kz_sub = _kz_forward(eps_sub, kx)
    kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
    kx0n = kx0 / k0
    R_eff, T_eff, jones = _assemble_jones_farfield(
        Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc, kx0n, N)
    return orders, R_eff, T_eff, jones, n_glob



# ===========================================================================
# Multi-region (segmented) PMM -- arbitrary piecewise-constant 1-D profiles
# (the spectral-element analogue of rcwa_jones_1d_segments).  A region wall
# lands on every element boundary, so eps is exact per element (no Gibbs); the
# binary ridge/groove path is the 2-segment special case.
# ===========================================================================

def _segment_walls(period, widths):
    """Cumulative region walls ``[0, w0, w0+w1, ..., period]`` (metres) for the
    fractional ``widths`` (must sum to 1 within 1e-6); normalized to land
    EXACTLY on 0 and ``period``."""
    w = np.asarray(widths, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("segments: need at least one region.")
    if np.any(w <= 0.0):
        raise ValueError("segments: every width fraction must be > 0.")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"segments: width fractions must sum to 1 (got {float(w.sum()):.6f}).")
    edges = np.concatenate([[0.0], np.cumsum(w)])
    return edges / edges[-1] * period



def _segment_elem_bnds(period, widths, materials, n_el_per_region, grade):
    """``elem_bnds`` list ``[(x_left, x_right, material), ...]`` for N graded
    regions (``n_el_per_region`` elements each; a wall on every region boundary).

    The regions are laid out in REVERSED order: the PMM's nodal ``x`` orientation
    is mirrored relative to the FMM (:func:`rcwa_jones_1d_segments`, which places
    ``segments[0]`` on ``x in [0, w0)``), and at oblique incidence the handedness
    matters -- reversing here makes the segmented PMM match the FMM order-by-order
    (verified; for a symmetric profile or normal incidence the order is
    immaterial, and the binary 2-region cell is always mirror-symmetric so the
    mirror was invisible there)."""
    widths = list(widths)[::-1]
    materials = list(materials)[::-1]
    walls = _segment_walls(period, widths)
    elem_bnds = []
    for i in range(len(widths)):
        b = _graded_boundaries(walls[i], walls[i + 1], n_el_per_region, grade)
        elem_bnds += list(zip(b[:-1], b[1:], [materials[i]] * n_el_per_region))
    # A single full-period element (one uniform region at n_el_per_region=1,
    # e.g. ``PMMStack.add_layer(eps=...)`` in an all-uniform stack) leaves the
    # periodic nodal basis too poor for the Rayleigh far-field match: energy
    # leaks ~2-30% into spurious orders while LOOKING plausible (bug found
    # 2026-06-10).  Splitting at the midpoint restores the proven >=2-element
    # grid -- Fresnel-exact on the uniform-film oracle, and identical to what
    # any patterned neighbour layer would force via the union grid anyway.
    if len(elem_bnds) == 1:
        xl, xr, mat = elem_bnds[0]
        xm = 0.5 * (xl + xr)
        elem_bnds = [(xl, xm, mat), (xm, xr, mat)]
    return elem_bnds



def _l2g_periodic(n_el, degree):
    """C0 local->global node map with the last node wrapped onto node 0
    (periodic); shared by the segmented assemblers."""
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    return l2g, last



def _build_sem_segments(period, widths, seg_eps, degree, n_el_per_region, grade):
    """Scalar N-region SEM operators (multi-region :func:`_build_sem`);
    ``seg_eps[i]`` = scalar permittivity of region ``i``."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    elem_bnds = _segment_elem_bnds(period, widths, seg_eps, n_el_per_region,
                                   grade)
    n_el = len(elem_bnds)
    l2g, n_glob = _l2g_periodic(n_el, degree)

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    S0, Peps, Pinv, L, Linv = _z(), _z(), _z(), _z(), _z()
    C, Cinv = _z(), _z()
    for e in range(n_el):
        xl, xr, eps = elem_bnds[e]
        J = 0.5 * (xr - xl)
        inv = 1.0 / eps
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Kloc = (Dphys.T * wel) @ Dphys
        Cloc = Mloc @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        S0[ix] += Mloc
        Peps[ix] += eps * Mloc
        Pinv[ix] += inv * Mloc
        L[ix] += Kloc
        Linv[ix] += inv * Kloc
        C[ix] += Cloc
        Cinv[ix] += inv * Cloc
    return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, C=C, Cinv=Cinv,
                n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds, degree=degree,
                ref_nodes=ref_nodes)



def _build_sem_tensor_segments(period, widths, seg_tensors, degree,
                               n_el_per_region, grade):
    """Tensor N-region SEM operators (multi-region :func:`_build_sem_tensor`);
    ``seg_tensors[i]`` = ``dict(exx, exy, eyx, eyy, ezz)`` of region ``i``."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    elem_bnds = _segment_elem_bnds(period, widths, seg_tensors, n_el_per_region,
                                   grade)
    n_el = len(elem_bnds)
    l2g, n_glob = _l2g_periodic(n_el, degree)

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    mass = {k: _z() for k in ("one", "eyy", "exy_xx", "eyx_xx", "schur", "ezz",
                              "inv_xx", "inv_ezz")}
    stiff = {k: _z() for k in ("one", "inv_ezz")}
    conv = {k: _z() for k in ("one", "inv_ezz")}
    for e in range(n_el):
        xl, xr, t = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Kloc = (Dphys.T * wel) @ Dphys
        Cloc = Mloc @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        iez = 1.0 / ezz
        mass["one"][ix] += Mloc
        mass["eyy"][ix] += eyy * Mloc
        mass["exy_xx"][ix] += (exy / exx) * Mloc
        mass["eyx_xx"][ix] += (eyx / exx) * Mloc
        mass["schur"][ix] += (eyy - eyx * exy / exx) * Mloc
        mass["ezz"][ix] += ezz * Mloc
        mass["inv_xx"][ix] += (1.0 / exx) * Mloc
        mass["inv_ezz"][ix] += iez * Mloc
        stiff["one"][ix] += Kloc
        stiff["inv_ezz"][ix] += iez * Kloc
        conv["one"][ix] += Cloc
        conv["inv_ezz"][ix] += iez * Cloc
    return dict(mass=mass, stiff=stiff, conv=conv, S0=mass["one"],
                n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds, degree=degree,
                ref_nodes=ref_nodes)



def _pmm_solve_segments(period, widths, seg_n, n_sub, n_sup, depth, wl, degree,
                        polarization, n_el_per_region, grade, far_field_orders,
                        angle=0.0):
    """Single-degree scalar PMM solve for an N-region profile (``seg_n[i]`` =
    refractive index of region ``i``)."""
    seg_eps = [_C(n) ** 2 for n in seg_n]
    eps_sup, eps_sub = _C(n_sup) ** 2, _C(n_sub) ** 2
    wl = _wood_safe_wl_1d(wl, angle, n_sup, period,
                          [eps_sup, eps_sub] + seg_eps, far_field_orders)
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(_C(n_sup))) * np.sin(float(angle)) * k0
    mats = _build_sem_segments(period, widths, seg_eps, degree,
                               n_el_per_region, grade)
    mats_sup = _build_sem_segments(period, widths, [eps_sup] * len(widths),
                                   degree, n_el_per_region, grade)
    mats_sub = _build_sem_segments(period, widths, [eps_sub] * len(widths),
                                   degree, n_el_per_region, grade)
    n_max = max([np.real(np.sqrt(e)) for e in seg_eps]
                + [np.real(_C(n_sup)), np.real(_C(n_sub))])
    return _pmm_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max,
                           period, depth, wl, degree, polarization,
                           far_field_orders, kx0,
                           label="pmm_efficiency_1d_segments", robust=True)



def _pmm_jones_solve_segments(period, widths, seg_tensors3, n_sub, n_sup, depth,
                              wl, degree, n_el_per_region, grade,
                              far_field_orders, angle=0.0):
    """Single-degree coupled anisotropic PMM solve for an N-region profile
    (``seg_tensors3[i]`` = the region's ``(3, 3)`` permittivity tensor)."""
    arrs = [np.asarray(M, dtype=_C) for M in seg_tensors3]

    def _t3(M):
        return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1],
                    ezz=M[2, 2])
    tensors = [_t3(M) for M in arrs]
    eps_sup, eps_sub = _C(n_sup) ** 2, _C(n_sub) ** 2
    wl = _wood_safe_wl_1d(
        wl, angle, n_sup, period,
        [eps_sup, eps_sub] + [M[i, i] for M in arrs for i in range(3)],
        far_field_orders)
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(_C(n_sup))) * np.sin(float(angle)) * k0
    mats = _build_sem_tensor_segments(period, widths, tensors, degree,
                                      n_el_per_region, grade)
    t_sup = dict(exx=eps_sup, exy=0.0, eyx=0.0, eyy=eps_sup, ezz=eps_sup)
    t_sub = dict(exx=eps_sub, exy=0.0, eyx=0.0, eyy=eps_sub, ezz=eps_sub)
    mats_sup = _build_sem_tensor_segments(period, widths, [t_sup] * len(widths),
                                          degree, n_el_per_region, grade)
    mats_sub = _build_sem_tensor_segments(period, widths, [t_sub] * len(widths),
                                          degree, n_el_per_region, grade)
    n_max = max([np.real(np.sqrt(M[0, 0])) for M in arrs]
                + [np.real(np.sqrt(M[1, 1])) for M in arrs]
                + [np.real(np.sqrt(eps_sup)), np.real(np.sqrt(eps_sub))])
    return _pmm_jones_solve_core(mats, mats_sup, mats_sub, eps_sup, eps_sub,
                                 n_max, period, depth, wl, degree,
                                 far_field_orders, kx0,
                                 label="pmm_jones_1d_segments", robust=True)



# --- shared stabilize (per-order convergence consensus) --------------------
def _energy_clean_pick(cluster, tots, target):
    """Pick the cluster member to return (v5.14 accuracy audit, the
    pseudo-plateau guard): when the structure is evidently LOSSLESS (some
    scanned solve conserves energy to ~1e-6), prefer the cluster member whose
    total is CLOSEST to the lossless target -- two marginal degrees can
    corroborate each other per-order within the cluster tolerance while both
    sit ~1e-3 off in energy (the measured pseudo-plateau returned per-order
    errors up to 1.5e-3 worse than lower degrees).  For lossy structures
    (no scanned solve near the target) energy is not a fitness signal, so the
    historical pick (the requested degree if clustered, else the first
    cluster member) is kept."""
    errs = [abs(t - target) for t in tots]
    if min(errs) < 1e-6 * max(target, 1.0):
        return min(cluster, key=lambda i: errs[i])
    return 0 if 0 in cluster else cluster[0]


class _StabilizeScanExhausted(Exception):
    """Raised by a ``solve_at_degree`` closure when the NEXT scan degree is
    unaffordable (e.g. the 2-D nodal-DOF cost cap): the scan ends gracefully
    and the consensus is evaluated on the degrees already solved."""


def _lossy_incidence(n_superstrate):
    """True when the INCIDENCE half-space absorbs (public ``Im(n_sup) > 0``).

    W7 F-A (2026-07-26).  ``sum(R)+sum(T) <= 1`` is a theorem only for a
    LOSSLESS incidence medium.  With an absorbing superstrate the family's
    ``kz_inc = Re(kz_sup)`` flux normalization legitimately returns totals
    ABOVE unity (measured on a plain slab: 1.00026 at ``Im(n_sup)=0.01``,
    1.0152 at 0.05, 1.0303 at 0.2 -- and rcwa/the Jones path agree to ~4e-7),
    so the stabilizers' super-unity gate is not a resonance discriminator
    there -- ``_require_propagating_incidence``, which every one of these
    entry points already calls, documents exactly this ("R + T != 1 by
    construction ... treat the sums as indicative"), so the gate was
    contradicting the contract stated one call earlier.  Measured pre-fix,
    ``pmm_efficiency_1d`` raised ``RuntimeError: no resonance-free
    solve in degrees [10, 26); the requested degree sits in a high-degree
    resonance band`` for EVERY degree from ``Im(n_sup) = 0.01`` up, blaming
    the user's degree for a perfectly healthy solve (the ``stabilize=False``
    answer matches rcwa to 4e-7).  This mirrors the skip already in
    :func:`~lumenairy.elements.pmm.twod._warn_lossless_energy_2d` and rcwa's
    ``_check_energy(..., lossless=)``.  A non-finite / non-scalar index reads
    False (keep the strict gate)."""
    try:
        return float(np.imag(_C(n_superstrate))) > 0.0
    except (TypeError, ValueError):        # traced / array-valued index
        return False


def _stabilize_scalar(solve_at_degree, d0, label, *, passive_tol=None,
                      per_order_tol=None, super_unity_ok=False):
    """Per-order convergence consensus over a degree window; ``solve_at_degree(d)
    -> (orders, R, T)``.  Shared by the binary + segmented scalar solvers (and
    the 2-D cell solver, whose closure maps the scan index onto odd degrees and
    raises :class:`_StabilizeScanExhausted` at its cost cap).  ``passive_tol`` /
    ``per_order_tol`` default to the 1-D-calibrated constants; the 2-D hybrid
    passes looser values (its Fourier-truncation energy floor ~1e-2 exceeds the
    1-D no-floor tolerance).

    ``super_unity_ok`` (W7 F-A) drops ONLY the ``tot <= 1 + tol`` half of the
    gate -- see :func:`_lossy_incidence`.  The lower bound and the per-order
    non-negativity stay, and the per-order convergence consensus (the real
    accuracy signal) is untouched."""
    passive_tol = _PASSIVE_TOL if passive_tol is None else passive_tol
    per_order_tol = _PER_ORDER_TOL if per_order_tol is None else per_order_tol
    scanned = []
    for d in range(d0, d0 + _STABILIZE_MAX_SCAN):
        try:
            orders, R, T = solve_at_degree(d)
        except _StabilizeScanExhausted:
            break
        tot = float(np.real(R.sum() + T.sum()))
        # TWO-SIDED passive gate + per-order non-negativity (audit P2-09):
        # the historical one-sided ``tot <= 1 + tol`` test certified grossly
        # NEGATIVE totals / per-order efficiencies -- a systematically-wrong
        # solve repeats itself at consecutive degrees and formed a bogus
        # 'converged cluster' with ZERO warnings.
        eff_min = min(float(np.min(np.real(R))) if np.size(R) else 0.0,
                      float(np.min(np.real(T))) if np.size(T) else 0.0)
        passive_ok = (-passive_tol <= tot
                      and (super_unity_ok or tot <= 1.0 + passive_tol)
                      and eff_min >= -passive_tol)
        scanned.append((d, orders, R, T, passive_ok, tot))
        records = [(s[1], (s[2], s[3]), None) for s in scanned]
        passive = [s[4] for s in scanned]
        cluster = _converged_cluster(records, passive, per_order_tol,
                                     _MIN_PLATEAU)
        if not cluster:
            continue
        pick = _energy_clean_pick(cluster, [s[5] for s in scanned], 1.0)
        return scanned[pick][1], scanned[pick][2], scanned[pick][3]
    passives = [s for s in scanned if s[4]]
    if not passives:
        raise RuntimeError(
            f"{label}: no resonance-free solve in degrees "
            f"[{d0}, {d0 + _STABILIZE_MAX_SCAN}); the requested degree sits in a "
            f"high-degree resonance band.  Use a lower degree (PMM converges "
            f"spectrally -- degree<=32 typically suffices) or "
            f"elements_per_region>1 with grade=True."
            + ("" if super_unity_ok else
               "  (If the incidence medium ABSORBS, sum R+T legitimately "
               "exceeds 1 under the Re(kz_inc) flux normalization -- that "
               "case is exempted from the super-unity gate.)"))
    warnings.warn(
        f"{label}: the per-order solution did not converge within degrees "
        f"[{d0}, {d0 + _STABILIZE_MAX_SCAN}); returning the highest degree tried "
        f"(degree {max(p[0] for p in passives)}).  It is likely UNDER-RESOLVED "
        f"(the total power can be passive while the per-order efficiencies are "
        f"still wrong) -- raise degree or use elements_per_region>1 with "
        f"grade=True.", stacklevel=3)
    best = max(passives, key=lambda s: s[0])
    return best[1], best[2], best[3]



def _stabilize_jones(solve_at_degree, d0, label, *, passive_tol=None,
                     per_order_tol=None, super_unity_ok=False):
    """Per-order + Jones convergence consensus; ``solve_at_degree(d) ->
    (orders, R, T, J)``.  Shared by the binary + segmented anisotropic solvers
    (and the 2-D tensor solver; see :class:`_StabilizeScanExhausted` and the
    tolerance / ``super_unity_ok`` notes on :func:`_stabilize_scalar` -- here
    the Jones target is 2, so the absorbing-superstrate exemption drops the
    ``tot <= 2 + 2 tol`` half of the gate)."""
    passive_tol = _JONES_PASSIVE_TOL if passive_tol is None else passive_tol
    per_order_tol = _PER_ORDER_TOL if per_order_tol is None else per_order_tol
    scanned = []
    for d in range(d0, d0 + _STABILIZE_MAX_SCAN):
        try:
            o, R, T, J = solve_at_degree(d)
        except _StabilizeScanExhausted:
            break
        tot = float(np.real(R.sum() + T.sum()))
        # TWO-SIDED + non-negative, as in _stabilize_scalar (audit P2-09);
        # the Jones twin's target is 2 (both incident pols) with the 2x tol.
        eff_min = min(float(np.min(np.real(R))) if np.size(R) else 0.0,
                      float(np.min(np.real(T))) if np.size(T) else 0.0)
        passive_ok = (-2.0 * passive_tol <= tot
                      and (super_unity_ok
                           or tot <= 2.0 + 2.0 * passive_tol)
                      and eff_min >= -2.0 * passive_tol)
        scanned.append((d, o, R, T, J, passive_ok, tot))
        records = [(s[1], (s[2], s[3]), s[4]) for s in scanned]
        passive = [s[5] for s in scanned]
        cluster = _converged_cluster(records, passive, per_order_tol,
                                     _MIN_PLATEAU)
        if not cluster:
            continue
        pick = _energy_clean_pick(cluster, [s[6] for s in scanned], 2.0)
        s = scanned[pick]
        return s[1], s[2], s[3], s[4]
    passives = [s for s in scanned if s[5]]
    if not passives:
        warnings.warn(
            f"{label}: no energy-passive solve in degrees "
            f"[{d0}, {d0 + _STABILIZE_MAX_SCAN}); returning the last attempt "
            f"(degree {d0 + _STABILIZE_MAX_SCAN - 1}).  It may sit in a resonance "
            f"band -- try a different degree or elements_per_region>1 with "
            f"grade=True.", stacklevel=3)
        s = scanned[-1]
        return s[1], s[2], s[3], s[4]
    warnings.warn(
        f"{label}: the per-order / Jones solution did not converge within "
        f"degrees [{d0}, {d0 + _STABILIZE_MAX_SCAN}); returning the highest "
        f"degree tried (degree {max(p[0] for p in passives)}).  It is likely "
        f"UNDER-RESOLVED (the total power can be passive while the per-order "
        f"split / Jones is still wrong) -- raise degree or use "
        f"elements_per_region>1 with grade=True.", stacklevel=3)
    best = max(passives, key=lambda s: s[0])
    return best[1], best[2], best[3], best[4]



# ===========================================================================
# JAX (differentiable) binary / normal-incidence path for pmm_efficiency_1d
# ===========================================================================
#
# A SELF-CONTAINED jax.numpy twin of the binary, NORMAL-incidence scalar PMM
# solve (_pmm_solve / _pmm_solve_core).  It is invoked ONLY when the index /
# geometry inputs are JAX arrays (see the dispatch in pmm_efficiency_1d); for
# NumPy inputs the original code runs verbatim, so the NumPy path is
# byte-identical by construction.  Scope is deliberately minimal (the
# de-risking spike): binary ridge/groove, angle == 0, fixed geometry, single
# fixed degree (stabilize=False semantics), elements_per_region == 1.
#
# The eps-INDEPENDENT topology (GLL nodes, the Lagrange derivative matrix, the
# local->global integer map, the per-element local mass / stiffness, AND the
# Rayleigh far-field projection T[m, i]) is precomputed ONCE in NumPy from the
# static (degree, duty, period, ...) integers and frozen as jnp constants -- so
# the trace carries only the eps / depth / wavelength dependence.  eps enters
# the Galerkin masses LINEARLY (Peps += eps*Mloc, Pinv += (1/eps)*Mloc), so the
# eps gradient flows once the in-place assembly is functionalised (.at[].add).
#
# The generalized modal pencil A x = q^2 B x (B = nodal mass S0 for TE, Pinv for
# TM) has NO JAX generalized-eig, so it is FOLDED to the standard problem
# eig(B^-1 A) -- B is well-conditioned here (the module's _safe_geig
# equilibration is not even triggered for well-scaled binary cells) -- and the
# differentiable custom-VJP eig (rcwa._jax_eig_stable, the torcwa/fmmax-style
# Lorentzian-broadened eigenvector VJP) is applied.  The fold reproduces the
# generalized spectrum to ~1e-12 (validated).  The forward-mode selector uses
# the NOISE-ROBUST branch (flip only when CLEARLY backward): a degenerate
# half-space q^2 carries ~1e-15 imaginary noise whose SIGN differs between the
# scipy-QZ generalized eig and the jnp folded eig, so the naive Im(q) < 0 test
# would flip propagating modes inconsistently and break the layer<->region
# pairing in the interface S-matrix (observed: R+T blew up to ~18).  The robust
# branch is forward-identical to the NumPy binary solve on the validated cells.


def _graded_fractions(n_el, grade):
    """The dimensionless element-boundary fractions s in [0, 1] (n_el+1 values)
    mirroring :func:`_graded_boundaries` -- the d_wall-INDEPENDENT part of the
    boundary map.  Physical boundaries are then a + (b - a) * s, which is a
    SMOOTH (linear) function of the endpoints, hence differentiable in d_wall.
    """
    if n_el <= 1:
        return np.array([0.0, 1.0])
    if not grade:
        return np.linspace(0.0, 1.0, n_el + 1)
    i = np.arange(n_el + 1)
    return 0.5 * (1.0 - np.cos(np.pi * i / n_el))



def _jpmm_build_topology(degree, n_ridge_el, n_groove_el, grade):
    """The truly-STATIC spectral-element topology -- depends ONLY on
    degree / element counts / grading, NOT on d_wall.  Returns the GLL
    reference nodes/weights, the reference derivative matrix, the local->global
    map, the per-region element fractions, and the region ids.  This is frozen
    as NumPy constants; the d_wall-dependent geometry (boundaries, Jacobians,
    masses, stiffnesses, projection phases) is rebuilt by
    :func:`_jpmm_build_dynamic` so the duty-cycle gradient flows."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    rfrac = _graded_fractions(n_ridge_el, grade)   # ridge boundary fractions
    gfrac = _graded_fractions(n_groove_el, grade)  # groove boundary fractions
    n_el = n_ridge_el + n_groove_el
    region = np.array([0] * n_ridge_el + [1] * n_groove_el, dtype=int)
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n_glob = last
    return dict(l2g=l2g, n_glob=n_glob, n_el=n_el, degree=degree,
                n_ridge_el=n_ridge_el, n_groove_el=n_groove_el,
                ref_nodes=ref_nodes, ref_w=ref_w, Dref=Dref,
                rfrac=rfrac, gfrac=gfrac, region=region)



def _jpmm_build_static(period, d_wall, degree, n_ridge_el, n_groove_el, grade):
    """Precompute the eps-INDEPENDENT spectral-element topology + per-element
    local mass / stiffness (all NumPy; frozen as jnp constants downstream).

    Mirrors :func:`_build_sem` but factors out the eps-weighted assembly so the
    differentiable path can rebuild S0 / Peps / Pinv / L / Linv functionally
    from traced ``eps_ridge`` / ``eps_groove``.  Region id 0 = ridge, 1 =
    groove.

    Kept as the NumPy (concrete, non-duty-traced) path: it composes the static
    topology with the concrete d_wall geometry.  The duty-traced path uses
    :func:`_jpmm_build_topology` + :func:`_jpmm_build_dynamic` instead so the
    Jacobians / masses / stiffnesses / projection phases carry the d_wall
    gradient."""
    topo = _jpmm_build_topology(degree, n_ridge_el, n_groove_el, grade)
    ref_w, Dref = topo["ref_w"], topo["Dref"]
    rb = 0.0 + (d_wall - 0.0) * topo["rfrac"]
    gb = d_wall + (period - d_wall) * topo["gfrac"]
    elem = ([(rb[i], rb[i + 1]) for i in range(n_ridge_el)]
            + [(gb[i], gb[i + 1]) for i in range(n_groove_el)])
    n_el = topo["n_el"]
    Mloc = np.zeros((n_el, degree + 1))
    Kloc = np.zeros((n_el, degree + 1, degree + 1))
    # Cloc = local convection mass INT phi phi' = diag(w*J) @ (Dref/J); the
    # weighted assembly (C = INT phi phi', Cinv = INT phi (1/eps) phi') drives
    # the oblique kx0 Bloch antisymmetrized convection -1j kx0 (C - C^T).  Its
    # J-scaling cancels (w*J)(Dref/J) = w*Dref, so it is d_wall-INDEPENDENT, but
    # kept per element for the assembly loop / the dynamic (duty-traced) twin.
    Cloc = np.zeros((n_el, degree + 1, degree + 1))
    for e, (xl, xr) in enumerate(elem):
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc[e] = wel
        Kloc[e] = (Dphys.T * wel) @ Dphys
        Cloc[e] = (wel[:, None] * Dphys)
    d = dict(topo)
    d.update(Mloc=Mloc, Kloc=Kloc, Cloc=Cloc,
             elem_bnds=[(xl, xr) for (xl, xr) in elem])
    return d



def _jpmm_build_dynamic(topo, jnp, period, d_wall):
    """Rebuild the d_wall-DEPENDENT spectral-element geometry in jnp so the
    duty-cycle gradient flows through the moving Jacobians.

    With the element COUNT held FIXED (the static topology), every global
    matrix entry is an ANALYTIC function of ``d_wall``: the element boundaries
    rb / gb are linear in d_wall, the Jacobian ``J = 0.5*(xr - xl)`` is linear,
    the local mass ``Mloc = ref_w * J`` scales as J, and the stiffness
    ``Kloc = (Dref/J)^T (ref_w*J) (Dref/J)`` scales as 1/J.  Returns jnp arrays
    ``Mloc`` (n_el, p+1), ``Kloc`` (n_el, p+1, p+1) and per-element endpoints
    ``xl`` / ``xr`` (n_el,) for the differentiable Fourier projection."""
    ref_w = jnp.asarray(topo["ref_w"])
    Dref = jnp.asarray(topo["Dref"])
    rfrac = jnp.asarray(topo["rfrac"])
    gfrac = jnp.asarray(topo["gfrac"])
    # Physical boundaries: linear (hence smooth) in d_wall.
    rb = d_wall * rfrac                              # [0, d_wall] graded
    gb = d_wall + (period - d_wall) * gfrac          # [d_wall, period] graded
    xl = jnp.concatenate([rb[:-1], gb[:-1]])
    xr = jnp.concatenate([rb[1:], gb[1:]])
    J = 0.5 * (xr - xl)                              # (n_el,)
    Mloc = ref_w[None, :] * J[:, None]               # (n_el, p+1)
    # Kloc[e] = (Dref/J).T @ diag(ref_w*J) @ (Dref/J) = (Dref.T @ diag(ref_w)
    # @ Dref) / J  -- assemble via einsum to keep it a clean traced tensor.
    Kref = jnp.einsum("ai,a,aj->ij", Dref, ref_w, Dref)   # (p+1, p+1), static-ish
    Kloc = Kref[None, :, :] / J[:, None, None]       # (n_el, p+1, p+1)
    # Cloc[e] = diag(ref_w*J) @ (Dref/J) = diag(ref_w) @ Dref -- the J-scaling
    # cancels, so the oblique convection mass is duty-INDEPENDENT; broadcast it
    # across elements to keep one tensor shape with the static path.
    Cref = ref_w[:, None] * Dref                     # (p+1, p+1)
    Cloc = jnp.broadcast_to(Cref[None, :, :], (J.shape[0],) + Cref.shape)
    return dict(Mloc=Mloc, Kloc=Kloc, Cloc=Cloc, xl=xl, xr=xr)



def _require_concrete_wavelength(wl, label, alt):
    """Reject a TRACED wavelength on a JAX PMM entry point.

    W7 F-E (2026-07-27).  The Rayleigh ORDER SET is chosen from the
    wavelength: :func:`_jpmm_order_set` sizes it as
    ``max(far_field_orders, 2*m_prop+5)`` with
    ``m_prop = floor(n_max * period / wl)`` -- a DATA-DEPENDENT INTEGER COUNT,
    and an integer count sets array SHAPES, which cannot be materialized from
    a tracer.  Under ``jax.jit`` / ``jax.grad`` the wavelength has no concrete
    value, so the old code silently fell back to ``wl = inf`` -> ``m_prop = 0``
    -> the order set COLLAPSED to the bare ``far_field_orders`` floor, DROPPING
    propagating orders that the NumPy policy includes.

    It was silent in the worst way: un-jitted the value is concrete, so the
    forward answer was bit-exact and only the TRACED evaluation was wrong.
    Measured pre-fix (2-layer stack, degree 24/30, n_sub 1.5, wl 633 nm,
    ``jax.jit`` over the wavelength):

    ========  ===  ======  =====  ============  ============
    period    ffo  NumPy N  jit N  forward rel   d/d(wl) rel
    ========  ===  ======  =====  ============  ============
    2.4 um      5      19      5     3.90e-02      1.76e-02
    3.2 um      5      25      5     4.15e-02      2.07e-01
    2.4 um      9      19      9     6.58e-15      9.14e-11
    3.2 um     11      25     11     5.74e-15      9.28e-09
    ========  ===  ======  =====  ============  ============

    -- a 20.7% wrong gradient and a 4.2% wrong forward, with ``jax.jit``
    returning a DIFFERENT ARRAY LENGTH than the un-jitted call on the same
    inputs (``(2, 5)`` vs ``(2, 9)``).  The last two rows are the control: once
    ``far_field_orders >= 2*m_prop+1`` there is nothing left to drop and the
    traced path is exact, which is why the default ``far_field_orders=21``
    hid this.

    Raising follows the EME precedent for a data-dependent selection under a
    trace.  A traced EPS is deliberately NOT rejected: it shrinks the order
    set the same way (n_max loses the traced component), but only EVANESCENT
    orders drop, so the totals stay exact -- measured 3.4e-15 / 6.7e-14 /
    2.9e-15 over the same periods.
    """
    try:                        # same probe the callers' _re_or_none uses
        float(np.real(np.asarray(wl)))
        return
    except (TypeError, ValueError):
        # JAX tracer errors (ConcretizationTypeError, TracerArrayConversion-
        # Error) subclass TypeError -- the exact carve-out used by the
        # berreman/eme JAX twins (except-budget discipline).
        pass
    raise NotImplementedError(
        f"{label}: a TRACED wavelength is not supported on the "
        f"differentiable path.  The propagating-order SET is selected from "
        f"the wavelength (m_prop = floor(n_max*period/wavelength)), and that "
        f"integer count fixes the result's array shapes -- it cannot be read "
        f"from a tracer, so jax.jit/jax.grad over the wavelength silently "
        f"solved a DIFFERENT, smaller order set than the NumPy path (measured "
        f"4.2% forward error and a 20.7% wrong d/d(wavelength), with jit and "
        f"un-jitted returning different array lengths).  At a FIXED concrete "
        f"wavelength the gradients are exact and fully supported -- eps "
        f"(re+im), layer thickness, incidence angle and the half-space "
        f"indices all match a NumPy central difference to <= 2e-8.  For a "
        f"dispersive / wavelength sweep use {alt}.")


def _jpmm_order_set(static, period, wl, n_max, far_field_orders, degree, label):
    """The Rayleigh order set for the forward far-field projection -- the SAME
    sizing as :func:`_pmm_solve_core`, computed from CONCRETE (real, static)
    numbers so the order COUNT (which sets the solve's array shapes) is fixed
    for a given jit trace."""
    n_glob = static["n_glob"]
    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: degree={degree} too low to resolve the "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
            f"degree or elements_per_region.")
    return np.arange(-half, half + 1)



def _jpmm_fourier_projection(orders, period, static):
    """``T[m, i]`` Rayleigh projection of the nodal Lagrange basis (NumPy; the
    static constant of the differentiable solve).  Identical quadrature to
    :func:`_sem_fourier_projection`."""
    from numpy.polynomial.legendre import leggauss
    l2g = static["l2g"]
    elem_bnds = static["elem_bnds"]
    degree = static["degree"]
    ref_nodes = static["ref_nodes"]
    n_glob = static["n_glob"]
    G = 2.0 * np.pi / period
    nq = max(2 * degree + 8, 24)
    xg, wg = leggauss(nq)
    wbary = np.ones(degree + 1)
    for j in range(degree + 1):
        for k in range(degree + 1):
            if k != j:
                wbary[j] /= (ref_nodes[j] - ref_nodes[k])

    def _lagrange_vals(xi):
        V = np.zeros((len(xi), degree + 1))
        for r, x in enumerate(xi):
            diff = x - ref_nodes
            if np.any(np.abs(diff) < 1e-14):
                V[r, np.argmin(np.abs(diff))] = 1.0
            else:
                num = wbary / diff
                V[r, :] = num / num.sum()
        return V

    Lv = _lagrange_vals(xg)
    T = np.zeros((len(orders), n_glob), dtype=_C)
    for e in range(len(elem_bnds)):
        xl, xr = elem_bnds[e]
        J = 0.5 * (xr - xl)
        xphys = 0.5 * (xr + xl) + J * xg
        phase = np.exp(-1j * np.outer(orders * G, xphys))
        contrib = (phase * (wg * J / period)) @ Lv
        idx = l2g[e]
        for a in range(degree + 1):
            T[:, idx[a]] += contrib[:, a]
    return T



def _jpmm_projection_quad(static):
    """The d_wall-INDEPENDENT quadrature pieces of the Fourier projection: the
    Gauss-Legendre reference nodes ``xg`` / weights ``wg`` and the Lagrange
    nodal-basis values ``Lv`` (nq, p+1) sampled at those nodes.  All static
    (depend only on degree).  Returned for reuse by the duty-traced jnp
    projection :func:`_jpmm_fourier_projection_jax`."""
    from numpy.polynomial.legendre import leggauss
    degree = static["degree"]
    ref_nodes = static["ref_nodes"]
    nq = max(2 * degree + 8, 24)
    xg, wg = leggauss(nq)
    wbary = np.ones(degree + 1)
    for j in range(degree + 1):
        for k in range(degree + 1):
            if k != j:
                wbary[j] /= (ref_nodes[j] - ref_nodes[k])
    Lv = np.zeros((nq, degree + 1))
    for r, x in enumerate(xg):
        diff = x - ref_nodes
        if np.any(np.abs(diff) < 1e-14):
            Lv[r, np.argmin(np.abs(diff))] = 1.0
        else:
            num = wbary / diff
            Lv[r, :] = num / num.sum()
    return xg, wg, Lv



def _jpmm_fourier_projection_jax(orders, period, static, jnp, dyn, quad):
    """``T[m, i]`` Rayleigh projection rebuilt in jnp from the traced per-element
    endpoints (``dyn['xl']`` / ``dyn['xr']``), so the duty-cycle gradient flows
    through the projection PHASES (x = 0.5*(xr+xl) + J*xi depends on d_wall).

    Algebra-identical to :func:`_jpmm_fourier_projection`; only the duty-moving
    physical node positions are traced, while the quadrature (xg, wg, Lv) and
    the global index map stay static."""
    cj = jnp.complex128
    l2g = static["l2g"]
    degree = static["degree"]
    n_glob = static["n_glob"]
    n_el = static["n_el"]
    xg, wg, Lv = quad
    xg = jnp.asarray(xg)
    wg = jnp.asarray(wg)
    Lv = jnp.asarray(Lv, cj)
    G = 2.0 * np.pi / period
    orders_j = jnp.asarray(orders)
    xl = dyn["xl"]
    xr = dyn["xr"]
    T = jnp.zeros((len(orders), n_glob), cj)
    for e in range(n_el):
        Je = 0.5 * (xr[e] - xl[e])
        xphys = 0.5 * (xr[e] + xl[e]) + Je * xg            # (nq,)
        phase = jnp.exp(-1j * jnp.outer(orders_j * G, xphys))   # (M, nq)
        contrib = (phase * (wg * Je / period)) @ Lv        # (M, p+1)
        idx = l2g[e]
        for a in range(degree + 1):
            T = T.at[:, idx[a]].add(contrib[:, a])
    return T



def _jpmm_assemble(static, jnp, eps_ridge, eps_groove, dyn=None):
    """Functionally assemble S0 / Peps / Pinv / L / Linv in jnp from the static
    per-element operators and the (traced) ridge / groove permittivities (eps
    enters the masses linearly).  Normal incidence -> no convection / kx0^2.

    When ``dyn`` (a :func:`_jpmm_build_dynamic` result) is supplied, the
    duty-traced per-element ``Mloc`` / ``Kloc`` are used instead of the frozen
    numpy ones, so the duty-cycle gradient flows through the moving Jacobians.
    """
    cj = jnp.complex128
    n_glob = static["n_glob"]
    n_el = static["n_el"]
    l2g = static["l2g"]
    region = static["region"]
    if dyn is None:
        Mloc = jnp.asarray(static["Mloc"], cj)
        Kloc = jnp.asarray(static["Kloc"], cj)
        Cloc = jnp.asarray(static["Cloc"], cj)
    else:
        Mloc = jnp.asarray(dyn["Mloc"], cj)
        Kloc = jnp.asarray(dyn["Kloc"], cj)
        Cloc = jnp.asarray(dyn["Cloc"], cj)
    eps_of = [eps_ridge, eps_groove]
    S0 = jnp.zeros((n_glob, n_glob), cj)
    Peps = jnp.zeros_like(S0)
    Pinv = jnp.zeros_like(S0)
    L = jnp.zeros_like(S0)
    Linv = jnp.zeros_like(S0)
    # Convection masses for the oblique kx0 Bloch shift (cheap; harmless at
    # normal incidence where the -1j kx0 (C - C^T) + kx0^2 mass terms vanish).
    C = jnp.zeros_like(S0)
    Cinv = jnp.zeros_like(S0)
    for e in range(n_el):
        eps = eps_of[region[e]]
        inv = 1.0 / eps
        idx = l2g[e]
        Ml = jnp.diag(Mloc[e])
        Kl = Kloc[e]
        Cl = Cloc[e]
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        S0 = S0.at[ii, jj].add(Ml)
        Peps = Peps.at[ii, jj].add(eps * Ml)
        Pinv = Pinv.at[ii, jj].add(inv * Ml)
        L = L.at[ii, jj].add(Kl)
        Linv = Linv.at[ii, jj].add(inv * Kl)
        C = C.at[ii, jj].add(Cl)
        Cinv = Cinv.at[ii, jj].add(inv * Cl)
    return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, C=C, Cinv=Cinv)



def _jpmm_sem_modes(M, jnp, eig, k0, polarization, kx0=0.0):
    """Folded standard-eig modal solve eig(B^-1 A) with the differentiable
    custom-VJP eig and the noise-robust forward-mode branch.  Returns
    ``(Acoef, lam, q, invop)`` (mirror of :func:`_sem_modes`).

    ``kx0`` (a TRACED jnp scalar = ``Re(n_sup) sin(angle) k0``) adds the Bloch
    shift of the pseudo-periodic envelope: the stiffness picks up the
    ANTISYMMETRIZED convection ``-1j kx0 (C - C^T)`` (the 1/eps-weighted
    ``Cinv`` for TM) and the ``kx0^2`` mass -- the exact transcription of the
    NumPy :func:`_sem_modes` oblique branch.  A ``kx0`` that is identically the
    python ``0.0`` (normal incidence) skips the convection so the path stays
    byte-equal to the prior normal-incidence twin; a traced kx0 (even one whose
    concrete value is 0) flows the ``d/d(angle)`` derivative through."""
    k02 = k0 * k0
    # Skip the convection ONLY for the python literal 0.0 (normal incidence,
    # byte-equal to the prior twin); a TRACED jnp scalar -- even one valued 0 --
    # is NOT a python float, so its d/d(angle) derivative flows.
    oblique = not (isinstance(kx0, float) and kx0 == 0.0)
    if polarization == "te":
        Lop = M["L"]
        if oblique:
            Cas = M["C"] - M["C"].T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * M["S0"]
        A, B = M["Peps"] - Lop / k02, M["S0"]
        invop = None
    else:
        Lop = M["Linv"]
        if oblique:
            Cas = M["Cinv"] - M["Cinv"].T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * M["Pinv"]
        A, B = M["S0"] - Lop / k02, M["Pinv"]
        invop = jnp.linalg.solve(M["S0"], M["Pinv"])
    q2, Acoef = eig(jnp.linalg.solve(B, A))
    q = jnp.sqrt(q2)
    # Noise-robust forward branch (the _sem_modes robust=True rule): flip only
    # when CLEARLY backward -- a degenerate half-space q^2 carries ~1e-15 imag
    # noise whose sign differs between scipy-QZ and the jnp folded eig.
    q = _forward_branch_flip(q, jnp)              # shared selector (S1-8)
    lam = -1j * q
    return Acoef, lam, q, invop



def _jpmm_solve(static, orders, Tp, jnp, eig, period, eps_ridge, eps_groove,
                eps_sup, eps_sub, depth, wl, polarization, dyn=None, kx0=0.0):
    """Self-contained jnp binary scalar PMM solve.  Returns ``(orders, R, T)``
    as jnp arrays, differentiable w.r.t. the permittivities, ``depth``, ``wl``
    AND -- at oblique incidence -- the incident ``angle`` / ``n_superstrate``
    (both threaded through the TRACED ``kx0``).  Algebra-identical to
    :func:`_pmm_solve_core`.

    ``kx0`` is the (traced) incident transverse wavenumber
    ``Re(n_sup) sin(angle) k0``; the python literal ``0.0`` selects the
    normal-incidence byte-equal branch.  At oblique it Bloch-shifts the modal
    operator (:func:`_jpmm_sem_modes`), the per-order ``kx = (kx0 + mG)/k0``,
    and the incident-flux normaliser carries the ``Ez`` component (the term
    that makes TM conserve at oblique).

    When ``dyn`` is supplied (a :func:`_jpmm_build_dynamic` result), the
    duty-moving Jacobians drive ALL three mass assemblies (layer + both
    half-spaces share ONE mesh, so the [W;V] interface continuity is consistent)
    and the duty-cycle gradient flows through the geometry."""
    cj = jnp.complex128
    k0 = 2.0 * jnp.pi / wl
    G = 2.0 * np.pi / period
    kx = (kx0 + jnp.asarray(orders) * G) / k0    # oblique: kx_m = (kx0+mG)/k0

    M = _jpmm_assemble(static, jnp, eps_ridge, eps_groove, dyn=dyn)
    Msup = _jpmm_assemble(static, jnp, eps_sup, eps_sup, dyn=dyn)
    Msub = _jpmm_assemble(static, jnp, eps_sub, eps_sub, dyn=dyn)

    Acoef, lam_l, q_l, invop = _jpmm_sem_modes(M, jnp, eig, k0, polarization,
                                               kx0)
    Wl = Acoef
    Vl = (Acoef if polarization == "te" else invop @ Acoef) @ jnp.diag(q_l)
    Wsup, _ls, q_sup, invsup = _jpmm_sem_modes(Msup, jnp, eig, k0, polarization,
                                               kx0)
    Wsub, _lb, q_sub, invsub = _jpmm_sem_modes(Msub, jnp, eig, k0, polarization,
                                               kx0)
    if polarization == "te":
        Vsup, Vsub = Wsup @ jnp.diag(q_sup), Wsub @ jnp.diag(q_sub)
    else:
        Vsup = (invsup @ Wsup) @ jnp.diag(q_sup)
        Vsub = (invsub @ Wsub) @ jnp.diag(q_sub)

    def _ismat(Wa, Va, Wb, Vb):
        a = jnp.linalg.solve(Wb, Wa)
        b = jnp.linalg.solve(Vb, Va)
        apb, amb = a + b, a - b
        iapb = jnp.linalg.inv(apb)
        return (-iapb @ amb, 2.0 * iapb,
                0.5 * (apb - amb @ iapb @ amb), amb @ iapb)

    def _psmat(lam, k0_L):
        n = lam.shape[0]
        X = jnp.diag(jnp.exp(-lam * k0_L))
        Z = jnp.zeros((n, n), cj)
        return (Z, X, X, Z)

    def _star(SA, SB):
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        n = A11.shape[0]
        I = jnp.eye(n, dtype=cj)
        D = jnp.linalg.inv(I - B11 @ A22)
        F = jnp.linalg.inv(I - A22 @ B11)
        return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
                B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)

    S = _ismat(Wsup, Vsup, Wl, Vl)
    S = _star(S, _psmat(lam_l, k0 * depth))
    S = _star(S, _ismat(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    Hsup = Tp @ Wsup
    Hsub = Tp @ Wsub
    delta0 = jnp.asarray((orders == 0).astype(_C))
    cinc = jnp.linalg.lstsq(Hsup, delta0, rcond=None)[0]
    r_ord = Hsup @ (S11 @ cinc)
    t_ord = Hsub @ (S21 @ cinc)

    def _kzf(eps, kxv):
        val = jnp.sqrt(jnp.asarray(eps - kxv ** 2, dtype=cj))
        return jnp.where(val.imag < 0.0, -val, val)

    kz_sup = _kzf(eps_sup, kx)
    kz_sub = _kzf(eps_sub, kx)
    # kz_inc stays a TRACED jnp scalar so d/d(wavelength) AND d/d(angle) /
    # d/d(n_sup) flow through the efficiency normaliser.  At oblique the
    # incident transverse wavenumber is kx0/k0 (the order-0 Rayleigh channel);
    # the scalar TM normaliser is the family-consistent incident flux in the
    # Hy gauge -- see the W7 F-B note on the NumPy `_scalar_farfield_RT`.  It is
    # written BRANCH-FREE here (a traced eps_sup cannot be tested for exact-real
    # loss), and reduces IDENTICALLY to kz_inc/eps_sup for a real superstrate.
    kx0n = kx0 / k0
    kz0 = _kzf(eps_sup, jnp.asarray(kx0n, cj))
    kz_inc = jnp.real(kz0)
    if polarization == "te":
        R = jnp.real(kz_sup / kz_inc) * jnp.abs(r_ord) ** 2
        T = jnp.real(kz_sub / kz_inc) * jnp.abs(t_ord) ** 2
    else:
        flux_inc = (((kz_inc ** 2 + kx0n ** 2) / kz_inc)
                    * jnp.abs(kz0 / eps_sup) ** 2)
        R = jnp.real(kz_sup / eps_sup) * jnp.abs(r_ord) ** 2 / flux_inc
        T = jnp.real(kz_sub / eps_sub) * jnp.abs(t_ord) ** 2 / flux_inc
    # ``Re(kz) > 0`` cut-off mask, matching the NumPy twin (audit M7).
    R = jnp.where(jnp.real(kz_sup) > 0.0, jnp.real(R), 0.0)
    T = jnp.where(jnp.real(kz_sub) > 0.0, jnp.real(T), 0.0)
    return orders, R, T



def _jpmm_concrete_incidence_guard(label, n_superstrate, angle):
    """Concrete-only mirror of the NumPy 1-D incidence-medium guard (the rcwa
    audit-P1 ``_require_propagating_incidence``, mirrored into the 1-D PMM by
    the 2026-07 wave-1 fix): reject a GAIN (public ``Im(n_superstrate) < 0``)
    or non-propagating (grazing / evanescent / metallic) incidence medium with
    the SAME ``ValueError`` the NumPy path raises -- the forward-root
    convention would flip ``kz_inc`` negative and silently negate every
    efficiency (NaN gradients downstream).

    Runs ONLY when BOTH ``n_superstrate`` and ``angle`` have concrete host
    values.  A TRACED value (a ``jax.grad`` / ``jax.jit`` Tracer) cannot be
    materialised without severing the trace (the rcwa
    ``_reject_jax_offplane`` tracer contract), so it SKIPS the guard: the JAX
    path guards only CONCRETE incidence media -- differentiating w.r.t.
    ``n_superstrate`` or ``angle`` bypasses the raise (documented in the
    INCIDENCE-MEDIUM SCOPE note on :func:`pmm_efficiency_1d`)."""
    try:
        nsup_c = complex(np.asarray(n_superstrate))
        angle_c = float(np.real(np.asarray(angle)))
    except Exception:               # Tracer: no concrete value -> skip
        return
    kx0n = float(np.real(nsup_c)) * np.sin(angle_c)
    _require_propagating_incidence(label, np.conj(_C(nsup_c ** 2)),
                                   kx0n ** 2)


def _pmm_efficiency_1d_jax(period, n_ridge, n_groove, n_substrate,
                           n_superstrate, depth, duty_cycle, wavelength,
                           *, angle, polarization, degree, elements_per_region,
                           grade, far_field_orders):
    """Differentiable (JAX) binary ``pmm_efficiency_1d`` (normal OR oblique).

    Invoked by :func:`pmm_efficiency_1d` when any index / geometry / angle
    input is a JAX array.  Supports the binary, fixed-single-degree surface at
    normal AND oblique incidence (real or complex/lossy eps); ``d/d(angle)`` and
    ``d/d(n_superstrate)`` flow through the TRACED Bloch wavenumber ``kx0``.
    Anything outside the surface (stabilize, multi-region) raises a precise
    error so the user is never silently given a wrong path.

    Oblique Wood-anomaly caveat: the propagating-order COUNT (array shapes) is
    sized from CONCRETE inputs and held static per trace, so ``d/d(angle)`` /
    ``d/d(wl)`` are valid only BETWEEN Rayleigh-order cutoffs.

    Incidence-medium guard: gain / non-propagating incidence media are
    rejected ONLY when ``n_superstrate`` and ``angle`` are concrete; a traced
    value skips the guard (see :func:`_jpmm_concrete_incidence_guard`)."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("pmm_efficiency_1d")

    if int(elements_per_region) != 1:
        raise NotImplementedError(
            "pmm_efficiency_1d: the JAX (differentiable) path currently "
            "supports elements_per_region == 1 only (the binary de-risking "
            "spike); use the NumPy path for hp-refinement.")

    # concrete-only incidence-medium guard (the wave-1 P1-03 NumPy mirror;
    # traced n_superstrate / angle skips it -- see the helper's docstring)
    _jpmm_concrete_incidence_guard("pmm_efficiency_1d", n_superstrate, angle)

    cj = jnp.complex128

    # CONCRETE (real, static) numbers for the shape-determining order COUNT,
    # resolved from the ORIGINAL arguments BEFORE the jnp.asarray promotion --
    # the COUNT must be fixed for a given trace (it sets every array shape).
    # When a sizing input is the differentiated variable it arrives as an
    # abstract tracer (no concrete value); _re_or_none then drops it and the
    # order count falls back to the remaining concrete inputs + the
    # far_field_orders floor.  This holds the count fixed across the trace --
    # eps / depth / wl flow through the solve body (shape-invariant) -- which is
    # the documented Wood-anomaly caveat (gradients are valid only BETWEEN order
    # cutoffs; do not differentiate across a Rayleigh anomaly that changes the
    # propagating-order count).
    def _re_or_none(v):
        try:
            return float(np.real(np.asarray(v)))
        except Exception:
            return None

    period_c = float(period)
    # W7 F-E: a traced wavelength cannot size the order set -- raise instead of
    # silently collapsing it to the far_field_orders floor.
    _require_concrete_wavelength(
        wavelength, "pmm_efficiency_1d",
        "pmm_efficiency_1d_vs_wavelength (or loop concrete wavelengths)")
    wl_c = _re_or_none(wavelength)
    n_max_vals = [v for v in (_re_or_none(n_superstrate), _re_or_none(n_substrate),
                              _re_or_none(n_ridge), _re_or_none(n_groove))
                  if v is not None]
    n_max = max(n_max_vals) if n_max_vals else 1.0

    # A CONCRETE duty value sizes the static topology + order set (shape-fixing).
    # When duty_cycle is the traced grad variable it arrives as an abstract
    # tracer; the topology (element COUNT, l2g, region) is duty-INDEPENDENT, so
    # any concrete proxy keeps shapes static while the d_wall-dependent geometry
    # (Jacobians, masses, projection phases) is rebuilt below in jnp from the
    # TRACED d_wall (Route B: smooth fixed-topology moving mesh).
    duty_traced = is_jax_array(duty_cycle)
    duty_c = _re_or_none(duty_cycle)
    if duty_c is None:
        duty_c = 0.5                                   # shape-only proxy
    if not (0.0 < duty_c < 1.0):
        raise ValueError(
            "pmm_efficiency_1d: the JAX (differentiable) path needs a strictly "
            f"interior duty_cycle (0 < duty < 1), got {duty_c}; a zero-width "
            "ridge or groove collapses an element (singular Jacobian).")
    d_wall_c = duty_c * period_c
    # The static topology depends ONLY on degree / element count / grading.
    static = _jpmm_build_static(period_c, d_wall_c, int(degree), 1, 1,
                                bool(grade))
    orders = _jpmm_order_set(static, period_c, wl_c, n_max,
                             int(far_field_orders), int(degree),
                             "pmm_efficiency_1d")

    n_ridge = jnp.asarray(n_ridge, cj)
    n_groove = jnp.asarray(n_groove, cj)
    n_substrate = jnp.asarray(n_substrate, cj)
    n_superstrate = jnp.asarray(n_superstrate, cj)
    depth = jnp.asarray(depth)
    wavelength = jnp.asarray(wavelength)
    eps_ridge = n_ridge ** 2
    eps_groove = n_groove ** 2
    eps_sup = n_superstrate ** 2
    eps_sub = n_substrate ** 2

    # Oblique Bloch wavenumber kx0 = Re(n_sup) sin(angle) k0 -- a TRACED jnp
    # scalar (function of angle AND Re(n_superstrate)) so d/d(angle) and
    # d/d(n_superstrate) flow.  At a concrete angle == 0 (and NOT traced) we
    # pass the python literal 0.0 so the solve takes the byte-equal
    # normal-incidence branch (no convection / kx0^2 / Ez normaliser shift) --
    # the Phase 0-2 path is untouched.  A TRACED angle (even one whose value is
    # 0) flows the convection so jax.grad(... , angle) is valid.
    angle_traced = is_jax_array(angle)
    if angle_traced or float(angle) != 0.0:
        k0_kx = 2.0 * jnp.pi / wavelength
        kx0 = jnp.real(n_superstrate) * jnp.sin(jnp.asarray(angle)) * k0_kx
    else:
        kx0 = 0.0

    if duty_traced:
        # Route B: rebuild the moving geometry in jnp so the duty gradient flows
        # through the Jacobians (masses / stiffnesses) AND the projection phases.
        d_wall = jnp.asarray(duty_cycle) * period_c
        topo = _jpmm_build_topology(int(degree), 1, 1, bool(grade))
        dyn = _jpmm_build_dynamic(topo, jnp, period_c, d_wall)
        quad = _jpmm_projection_quad(static)
        Tp = _jpmm_fourier_projection_jax(orders, period_c, static, jnp, dyn,
                                          quad)
    else:
        dyn = None
        Tp = jnp.asarray(_jpmm_fourier_projection(orders, period_c, static), cj)

    eig = _jax_eig_stable()
    o, R, T = _jpmm_solve(static, orders, Tp, jnp, eig, period_c,
                          eps_ridge, eps_groove, eps_sup, eps_sub,
                          depth, wavelength, polarization, dyn=dyn, kx0=kx0)
    return jnp.asarray(orders), R, T



# ===========================================================================
# JAX (differentiable) Jones path for pmm_jones_1d -- Phase 4
# ===========================================================================
#
# A SELF-CONTAINED jax.numpy twin of the coupled ANISOTROPIC (in-plane 2x2-
# tensor) binary Jones solve (_pmm_jones_solve / _pmm_jones_solve_core /
# _sem_modes_tensor), differentiable w.r.t. the IN-PLANE tensor entries
# (exx, exy, eyx, eyy, ezz), depth, wavelength, angle and the half-space
# indices.  Invoked ONLY when an input is a JAX array (the dispatch in
# pmm_jones_1d); NumPy inputs run the original code verbatim, so the NumPy path
# is byte-identical by construction.  Scope (the validated Jones surface):
# binary ridge/groove, IN-PLANE tensor (exz/eyz/ezx/ezy == 0), VERTICAL wall
# (slant == 0), NORMAL or OBLIQUE incidence, real OR complex/lossy eps, a single
# fixed degree (stabilize=False semantics), elements_per_region == 1.
#
# Unlike the scalar generalized pencil A x = q^2 B x (folded to eig(B^-1 A)),
# the Jones modal solver is a STANDARD eig of the 2n x 2n coupled [Ex; Ey] block
# Mbig (np.linalg.eig(Mbig) directly), so the rcwa custom-VJP eig applies
# DIRECTLY -- no generalized fold.  The tensor enters the Galerkin masses
# LINEARLY in eps (the wall-normal inverse-rule block Cxx = [[1/exx]]^-1 is the
# only nonlinearity, an inv of an eps-linear mass), so the tensor-entry gradient
# (incl off-diagonal exy/eyx -- the cross-pol coupling) flows.  The forward set
# is chosen by the z-Poynting flux with the DIFFERENTIABLE noise-robust
# jnp.where (NO argsort / compaction).  At NORMAL incidence the degenerate
# Ex/Ey-dominant modes are mixed by np.linalg.eig in the degenerate subspace,
# but the Jones OBSERVABLES (|J|^2, efficiencies) are GAUGE-INVARIANT, so the
# in-subspace rotation does not affect them -- no argmax / sort / phase gauge
# fix is applied (it would corrupt the gauge-invariant grads).


def _jpmm_assemble_tensor(static, jnp, t_ridge, t_groove, dyn=None):
    """Functionally assemble the per-coefficient tensor Galerkin masses
    (``one``/``eyy``/``exy_xx``/``eyx_xx``/``schur``/``inv_xx``/``inv_ezz``) and
    the ``one``/``inv_ezz`` weighted stiffnesses + convection masses in jnp from
    the static per-element local operators and the (traced) ridge / groove
    tensor dicts ``dict(exx, exy, eyx, eyy, ezz)`` -- the differentiable twin of
    :func:`_build_sem_tensor`.  Every coefficient enters its mass LINEARLY, so
    the tensor-entry gradient flows (the only nonlinearity, ``Cxx =
    [[1/exx]]^-1``, is an inv applied later in :func:`_jpmm_sem_modes_tensor`).

    ``dyn`` (a :func:`_jpmm_build_dynamic` result) swaps in the duty-traced
    per-element ``Mloc`` / ``Kloc`` / ``Cloc`` (not used by the Jones surface
    yet -- duty is held fixed -- but kept symmetric with the scalar path)."""
    cj = jnp.complex128
    n_glob = static["n_glob"]
    n_el = static["n_el"]
    l2g = static["l2g"]
    region = static["region"]
    if dyn is None:
        Mloc = jnp.asarray(static["Mloc"], cj)
        Kloc = jnp.asarray(static["Kloc"], cj)
        Cloc = jnp.asarray(static["Cloc"], cj)
    else:
        Mloc = jnp.asarray(dyn["Mloc"], cj)
        Kloc = jnp.asarray(dyn["Kloc"], cj)
        Cloc = jnp.asarray(dyn["Cloc"], cj)
    t_of = [t_ridge, t_groove]
    Z = jnp.zeros((n_glob, n_glob), cj)
    mass = {k: Z for k in ("one", "eyy", "exy_xx", "eyx_xx", "schur",
                           "inv_xx", "inv_ezz")}
    stiff = {k: Z for k in ("one", "inv_ezz")}
    conv = {k: Z for k in ("one", "inv_ezz")}
    for e in range(n_el):
        t = t_of[region[e]]
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        iez = 1.0 / ezz
        idx = l2g[e]
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        Ml = jnp.diag(Mloc[e])
        Kl = Kloc[e]
        Cl = Cloc[e]
        mass["one"] = mass["one"].at[ii, jj].add(Ml)
        mass["eyy"] = mass["eyy"].at[ii, jj].add(eyy * Ml)
        mass["exy_xx"] = mass["exy_xx"].at[ii, jj].add((exy / exx) * Ml)
        mass["eyx_xx"] = mass["eyx_xx"].at[ii, jj].add((eyx / exx) * Ml)
        mass["schur"] = mass["schur"].at[ii, jj].add(
            (eyy - eyx * exy / exx) * Ml)
        mass["inv_xx"] = mass["inv_xx"].at[ii, jj].add((1.0 / exx) * Ml)
        mass["inv_ezz"] = mass["inv_ezz"].at[ii, jj].add(iez * Ml)
        stiff["one"] = stiff["one"].at[ii, jj].add(Kl)
        stiff["inv_ezz"] = stiff["inv_ezz"].at[ii, jj].add(iez * Kl)
        conv["one"] = conv["one"].at[ii, jj].add(Cl)
        conv["inv_ezz"] = conv["inv_ezz"].at[ii, jj].add(iez * Cl)
    return dict(mass=mass, stiff=stiff, conv=conv, S0=mass["one"],
                n_glob=n_glob)



def _jpmm_sem_modes_tensor(mats, jnp, eig, k0, kx0=0.0):
    """Coupled ``(E_x, E_y)`` anisotropic modal solve -- the differentiable twin
    of :func:`_sem_modes_tensor`.  Returns ``(W2, V2, lam, q)`` (``W2[:, n]`` =
    the 2-vector field of mode ``n`` stacked ``[Ex; Ey]``; ``V2`` the matched
    magnetic partner ``Q W diag(1/lam)``; ``q = gamma/k0``; ``lam = -i q``).

    The 2n x 2n coupled block ``Mbig = [[G Cxx, G Cxy], [Cyx, Cyy - Kx2]]`` is a
    STANDARD eig (no generalized fold), so the rcwa custom-VJP ``eig`` applies
    directly.  The forward set is chosen by the z-POYNTING FLUX with a
    DIFFERENTIABLE noise-robust :func:`jnp.where` (flip a mode only when CLEARLY
    backward); at ``kx0 == 0`` the propagating modes are real so the flux split
    reduces to the legacy ``Im(q) >= 0`` rule (matched via the same robust
    fallback the scalar twin uses)."""
    cj = jnp.complex128
    n = mats["n_glob"]
    k02 = k0 * k0
    S0 = mats["S0"]
    mass, stiff, conv = mats["mass"], mats["stiff"], mats["conv"]

    iS0 = jnp.linalg.inv(S0)
    Cinv_xx = iS0 @ mass["inv_xx"]          # [[1/exx]]
    Cxx = jnp.linalg.inv(Cinv_xx)           # [[1/exx]]^-1 (wall-normal inv rule)
    EXY_XX = iS0 @ mass["exy_xx"]           # [[exy/exx]]
    EYX_XX = iS0 @ mass["eyx_xx"]           # [[eyx/exx]]
    SCHUR = iS0 @ mass["schur"]             # [[eyy - eyx exy/exx]]
    Cxy = Cxx @ EXY_XX
    Cyx = EYX_XX @ Cxx
    Cyy = SCHUR + EYX_XX @ Cxx @ EXY_XX

    # Skip the convection ONLY for the python literal 0.0 (normal incidence,
    # byte-equal to the prior twin); a TRACED jnp scalar -- even one valued 0 --
    # is NOT a python float, so its d/d(angle) derivative flows.
    oblique = not (isinstance(kx0, float) and kx0 == 0.0)

    def _kxop(skey, ckey, mkey):
        op = stiff[skey]
        if oblique:
            Cw = conv[ckey]
            op = op - 1j * kx0 * (Cw - Cw.T) + (kx0 * kx0) * mass[mkey]
        return (1.0 / k02) * (iS0 @ op)
    KxEzziKx = _kxop("inv_ezz", "inv_ezz", "inv_ezz")
    Kx2 = _kxop("one", "one", "one")
    G = jnp.eye(n, dtype=cj) - KxEzziKx

    Mbig = jnp.block([[G @ Cxx, G @ Cxy],
                      [Cyx,     Cyy - Kx2]])
    q2, W2 = eig(Mbig)
    q = jnp.sqrt(q2)
    Q = jnp.block([[Cyx, Cyy - Kx2], [-Cxx, -Cxy]])

    # POYNTING-FLUX forward selector (the _sem_modes_tensor non-legacy branch,
    # also valid at kx0 == 0 since the propagating modes are real there): the V2
    # partner is [Hx; Hy] and the modal H carries an extra -i, so
    # Sz_n = Im( Ex.S0.conj(Hy) - Ey.S0.conj(Hx) ).  Flip only when CLEARLY
    # backward (a degenerate q^2 carries ~1e-15 imag noise whose sign differs
    # between scipy-QZ and jnp eig -> the naive Im(q)<0 test would flip
    # propagating modes inconsistently).
    lam0 = -1j * q
    safe0 = jnp.where(jnp.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = Q @ W2 @ jnp.diag(1.0 / safe0)
    SVt = S0 @ jnp.conj(V0[:n])             # S0 conj(Hx)
    SVb = S0 @ jnp.conj(V0[n:])             # S0 conj(Hy)
    flux = jnp.imag(jnp.einsum("in,in->n", W2[:n], SVb)
                    - jnp.einsum("in,in->n", W2[n:], SVt))
    prop = _mass_flux_cut(flux, W2, SVt, SVb, n, jnp)  # W7 B2: unit-safe cut
    flip = jnp.where(prop, flux < 0.0, q.imag < 0.0)
    q = jnp.where(flip, -q, q)
    lam = -1j * q
    safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
    V2 = Q @ W2 @ jnp.diag(1.0 / safe)
    return W2, V2, lam, q


def _juniform_geo_eig(mats, jnp, eig, k0, kx0=0.0):
    """Eps-free GEOMETRIC eigendecomposition for a uniform isotropic medium --
    the differentiable twin of :func:`_uniform_geo_eig` (backlog A2).  For a
    uniform isotropic ``eps`` the coupled :func:`_jpmm_sem_modes_tensor`
    operator collapses EXACTLY to ``Mbig(eps) = eps I_(2n) - blockdiag(Kx2, Kx2)``,
    so ONE eig of the geometry-only ``Kx2`` serves every uniform medium on the
    shared nodal grid.  ``Kx2`` depends on the mesh + (traced) ``kx0`` only, NOT
    on ``eps`` -- so both half-spaces reuse it.  Returns ``(mu, w, Kx2)``."""
    k02 = k0 * k0
    iS0 = jnp.linalg.inv(mats["S0"])
    op = mats["stiff"]["one"]
    # traced kx0 (even valued 0) is not a python float -> keep the convection so
    # d/d(angle) flows, exactly as _jpmm_sem_modes_tensor does.
    oblique = not (isinstance(kx0, float) and kx0 == 0.0)
    if oblique:
        Cw = mats["conv"]["one"]
        op = op - 1j * kx0 * (Cw - Cw.T) + (kx0 * kx0) * mats["mass"]["one"]
    Kx2 = (1.0 / k02) * (iS0 @ op)
    mu, w = eig(Kx2)
    return mu, w, Kx2


def _jpmm_sem_modes_uniform(mats, jnp, eig, k0, kx0, eps, geo=None):
    """Uniform-isotropic modes from the SHARED geometric eig -- the
    differentiable twin of :func:`_sem_modes_uniform`, returning the IDENTICAL
    ``(W2, V2, lam, q)`` contract as :func:`_jpmm_sem_modes_tensor` with the
    same z-Poynting forward selector, evaluated on the eps-shifted spectrum
    ``q^2 = eps - mu`` (the branch choice legitimately depends on eps).  The
    block-diagonal eigenvector gauge differs from a raw 2n eig but is physically
    equivalent -- the downstream interface S-matrix is basis-agnostic."""
    cj = jnp.complex128
    n = mats["n_glob"]
    eps = jnp.asarray(eps, dtype=cj)
    if geo is None:
        geo = _juniform_geo_eig(mats, jnp, eig, k0, kx0)
    mu, w, _Kx2 = geo
    q2 = eps - mu
    q = jnp.sqrt(jnp.concatenate([q2, q2]))
    Z = jnp.zeros((n, n), dtype=cj)
    W2 = jnp.block([[w, Z], [Z, w]])
    # Q @ W2 for the uniform medium: (eps I - Kx2) w = w diag(q2), Kx2 w = w mu.
    QW = jnp.block([[Z, w * q2[None, :]], [-eps * w, Z]])
    S0 = mats["S0"]
    lam0 = -1j * q
    safe0 = jnp.where(jnp.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = QW * (1.0 / safe0)[None, :]
    SVt = S0 @ jnp.conj(V0[:n])
    SVb = S0 @ jnp.conj(V0[n:])
    flux = jnp.imag(jnp.einsum("in,in->n", W2[:n], SVb)
                    - jnp.einsum("in,in->n", W2[n:], SVt))
    prop = _mass_flux_cut(flux, W2, SVt, SVb, n, jnp)  # W7 B2: unit-safe cut
    flip = jnp.where(prop, flux < 0.0, q.imag < 0.0)
    q = jnp.where(flip, -q, q)
    lam = -1j * q
    safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
    V2 = QW * (1.0 / safe)[None, :]
    return W2, V2, lam, q


def _jpmm_jones_solve(static, orders, Tp, jnp, eig, period, t_ridge, t_groove,
                      eps_sup, eps_sub, depth, wl, kx0=0.0, dyn=None):
    """Self-contained jnp coupled anisotropic (Jones) PMM solve -- the
    differentiable twin of :func:`_pmm_jones_solve_core`.  Returns
    ``(orders, R(2,N), T(2,N), jones(2,2))`` as jnp arrays, differentiable
    w.r.t. the in-plane tensor entries, ``depth``, ``wl`` and (at oblique) the
    incident ``angle`` / ``n_superstrate`` (threaded through the TRACED ``kx0``).

    The half-spaces are ISOTROPIC tensors (``exx = eyy = ezz = eps``,
    ``exy = eyx = 0``); they share the layer's mesh so the ``[W; V]`` interface
    continuity is consistent.  Everything is built FUNCTIONALLY (jnp stacking,
    no in-place ``np.zeros`` mutation)."""
    cj = jnp.complex128
    k0 = 2.0 * jnp.pi / wl
    n_glob = static["n_glob"]
    G = 2.0 * np.pi / period
    N = len(orders)

    t_iso_sup = dict(exx=eps_sup, exy=0.0 * eps_sup, eyx=0.0 * eps_sup,
                     eyy=eps_sup, ezz=eps_sup)
    t_iso_sub = dict(exx=eps_sub, exy=0.0 * eps_sub, eyx=0.0 * eps_sub,
                     eyy=eps_sub, ezz=eps_sub)
    mats = _jpmm_assemble_tensor(static, jnp, t_ridge, t_groove, dyn=dyn)
    mats_sup = _jpmm_assemble_tensor(static, jnp, t_iso_sup, t_iso_sup, dyn=dyn)
    mats_sub = _jpmm_assemble_tensor(static, jnp, t_iso_sub, t_iso_sub, dyn=dyn)

    Wl, Vl, lam_l, _ql = _jpmm_sem_modes_tensor(mats, jnp, eig, k0, kx0)
    # v5.18.1 (audit P3-27 second half): the two ISOTROPIC half-spaces share
    # ONE geometry-only eig (backlog A2) instead of two independent full 2n
    # eigs -- Kx2 is eps-free and identical for sup/sub on the shared mesh.
    # Mirrors the numpy _pmm_jones_solve_core, which already does this; the JAX
    # twin now matches that oracle's shared-eig gauge exactly.
    _geo = _juniform_geo_eig(mats_sup, jnp, eig, k0, kx0)
    Wsup, Vsup, _ls, _qs = _jpmm_sem_modes_uniform(
        mats_sup, jnp, eig, k0, kx0, eps_sup, geo=_geo)
    Wsub, Vsub, _lb, _qb = _jpmm_sem_modes_uniform(
        mats_sub, jnp, eig, k0, kx0, eps_sub, geo=_geo)

    def _ismat(Wa, Va, Wb, Vb):
        a = jnp.linalg.solve(Wb, Wa)
        b = jnp.linalg.solve(Vb, Va)
        apb, amb = a + b, a - b
        iapb = jnp.linalg.inv(apb)
        return (-iapb @ amb, 2.0 * iapb,
                0.5 * (apb - amb @ iapb @ amb), amb @ iapb)

    def _psmat(lam, k0_L):
        m = lam.shape[0]
        X = jnp.diag(jnp.exp(-lam * k0_L))
        Z = jnp.zeros((m, m), cj)
        return (Z, X, X, Z)

    def _star(SA, SB):
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        m = A11.shape[0]
        I = jnp.eye(m, dtype=cj)
        D = jnp.linalg.inv(I - B11 @ A22)
        F = jnp.linalg.inv(I - A22 @ B11)
        return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
                B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)

    S = _ismat(Wsup, Vsup, Wl, Vl)
    S = _star(S, _psmat(lam_l, k0 * depth))
    S = _star(S, _ismat(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    def _project(Wmodes):
        return jnp.vstack([Tp @ Wmodes[:n_glob, :], Tp @ Wmodes[n_glob:, :]])

    Hsup = _project(Wsup)
    Hsub = _project(Wsub)

    orders_j = jnp.asarray(orders)
    kx = (kx0 + orders_j * G) / k0

    def _kzf(eps, kxv):
        val = jnp.sqrt(jnp.asarray(eps - kxv ** 2, dtype=cj))
        return jnp.where(val.imag < 0.0, -val, val)

    kz_sup = _kzf(eps_sup, kx)
    kz_sub = _kzf(eps_sub, kx)
    kx0n = kx0 / k0
    kz_inc = jnp.real(_kzf(eps_sup, jnp.asarray(kx0n, cj)))
    safe_r = jnp.where(jnp.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = jnp.where(jnp.abs(kz_sub) < 1e-12, 1.0, kz_sub)

    # DIFFERENTIABLE minimum-norm least squares for the incident-amplitude
    # projection.  ``jnp.linalg.lstsq``'s VJP NaNs on a rank-deficient / under-
    # determined system (the stacked Hsup is (2N, 2*n_glob) with 2N <= 2*n_glob,
    # i.e. underdetermined), so use the closed-form min-norm pseudo-inverse
    # x = A^H (A A^H)^-1 b (forward-identical to numpy's SVD min-norm lstsq to
    # ~1e-14, validated; A A^H is well-conditioned -- cond ~ Hsup's^2).  The
    # over-determined case (2N > 2*n_glob, not hit on this surface) would use
    # (A^H A)^-1 A^H; branch on the CONCRETE shape (static per trace).
    mrows, ncols = Hsup.shape
    if mrows <= ncols:
        AAH_inv = jnp.linalg.inv(Hsup @ Hsup.conj().T)
        pinv = Hsup.conj().T @ AAH_inv          # (ncols, mrows) min-norm pinv
    else:
        AHA_inv = jnp.linalg.inv(Hsup.conj().T @ Hsup)
        pinv = AHA_inv @ Hsup.conj().T          # least-squares pinv

    m0 = int(np.where(orders == 0)[0][0])
    rows_R, rows_T, jcols = [], [], []
    for col in range(2):                    # 0 = incident Ex, 1 = incident Ey
        rhs = jnp.zeros(2 * N, cj).at[(col * N) + m0].set(1.0)
        cinc = pinv @ rhs
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:N], r_ord[N:]
        tx, ty = t_ord[:N], t_ord[N:]
        # longitudinal Ez (div D = 0 in the isotropic half-space): rz = -kx rx/kz
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        # per-column incident flux: col-0 (Ex, p-pol) carries Ez_inc -> z-flux
        # kz_inc(1+(kx0/kz_inc)^2); col-1 (Ey, s-pol) -> kz_inc.  At kx0=0 both
        # reduce to kz_inc (byte-equal to normal incidence).
        if col == 0:
            flux_inc = kz_inc * (1.0 + (kx0n / kz_inc) ** 2)
        else:
            flux_inc = kz_inc
        Re = jnp.real(kz_sup) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                 + jnp.abs(rz) ** 2) / flux_inc
        Te = jnp.real(kz_sub) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                 + jnp.abs(tz) ** 2) / flux_inc
        # ``Re(kz) > 0`` cut-off mask, matching the NumPy twin (audit M7).
        rows_R.append(jnp.where(jnp.real(kz_sup) > 0.0, jnp.real(Re), 0.0))
        rows_T.append(jnp.where(jnp.real(kz_sub) > 0.0, jnp.real(Te), 0.0))
        jcols.append(jnp.stack([rx[m0], ry[m0]]))   # [Ex; Ey] reflected, order 0
    R_eff = jnp.stack(rows_R)
    T_eff = jnp.stack(rows_T)
    jones = jnp.stack(jcols, axis=1)        # (2,2): columns = incident Ex / Ey
    return orders, R_eff, T_eff, jones



def _pmm_jones_1d_jax(period, eps_ridge, eps_groove, n_substrate, n_superstrate,
                      depth, duty_cycle, wavelength, *, angle, degree,
                      elements_per_region, grade, far_field_orders):
    """Differentiable (JAX) binary ``pmm_jones_1d`` -- the full ``2x2`` Jones
    response of an IN-PLANE anisotropic binary grating (normal OR oblique,
    vertical wall).  Invoked by :func:`pmm_jones_1d` when any tensor / geometry /
    angle input is a JAX array.  Differentiable w.r.t. the in-plane tensor
    entries (incl off-diagonal ``exy`` / ``eyx``), ``depth``, ``wavelength``,
    ``angle`` and the half-space indices.  Anything outside the surface
    (stabilize, multi-region, out-of-plane, slant) raises (handled upstream).

    Incidence-medium guard: gain / non-propagating incidence media are
    rejected ONLY when ``n_superstrate`` and ``angle`` are concrete; a traced
    value skips the guard (see :func:`_jpmm_concrete_incidence_guard`)."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("pmm_jones_1d")

    if int(elements_per_region) != 1:
        raise NotImplementedError(
            "pmm_jones_1d: the JAX (differentiable) path currently supports "
            "elements_per_region == 1 only; use the NumPy path for "
            "hp-refinement.")

    # concrete-only incidence-medium guard (the wave-1 P1-03 NumPy mirror;
    # traced n_superstrate / angle skips it -- see the helper's docstring)
    _jpmm_concrete_incidence_guard("pmm_jones_1d", n_superstrate, angle)

    cj = jnp.complex128
    er = jnp.asarray(eps_ridge, cj)
    eg = jnp.asarray(eps_groove, cj)

    def _t3(M):
        return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1],
                    ezz=M[2, 2])
    t_ridge, t_groove = _t3(er), _t3(eg)

    # CONCRETE numbers for the shape-determining order COUNT (held static per
    # trace -- it sets every array shape).  A traced sizing input arrives as a
    # tracer with no concrete value; _re_or_none drops it and the count falls
    # back to the remaining concrete inputs + the far_field_orders floor (the
    # documented Wood-anomaly caveat: grads valid only BETWEEN order cutoffs).
    def _re_or_none(v):
        try:
            return float(np.real(np.asarray(v)))
        except Exception:
            return None

    period_c = float(period)
    # W7 F-E: see _require_concrete_wavelength.
    _require_concrete_wavelength(
        wavelength, "pmm_jones_1d",
        "pmm_jones_1d_vs_wavelength (or loop concrete wavelengths)")
    wl_c = _re_or_none(wavelength)
    nsup_c = _re_or_none(n_superstrate)
    nsub_c = _re_or_none(n_substrate)
    # n_max from the COMPLEX eps (Re(sqrt(eps))), matching the numpy reference
    # _pmm_jones_solve.  Stripping Im first then sqrt gives NaN for a metal/ENZ
    # eps (sqrt of a negative real -> int(NaN) crash in _n_propagating_orders) and
    # under-counts propagating orders for a lossy eps (Re(sqrt(1+12j))=2.55, not 1).
    def _n_or_none(v):
        try:
            return float(np.real(np.sqrt(np.asarray(v).astype(_C))))
        except Exception:
            return None
    n_max_vals = [v for v in (_n_or_none(er[0, 0]), _n_or_none(er[1, 1]),
                              _n_or_none(eg[0, 0]), _n_or_none(eg[1, 1]))
                  if v is not None]
    n_max_vals += [v for v in (nsup_c, nsub_c) if v is not None]
    n_max = max(n_max_vals) if n_max_vals else 1.0

    duty_c = _re_or_none(duty_cycle)
    if duty_c is None:
        # A traced duty_cycle would be SILENTLY frozen at 0.5 here (the static
        # topology + numpy projection bake d_wall = duty*period), returning the
        # duty=0.5 answer for every duty and a 0.0 duty-gradient.  Raise instead:
        # the moving-mesh duty rebuild exists only on the scalar
        # pmm_efficiency_1d path (Route-B), not the Jones surface.
        raise NotImplementedError(
            "pmm_jones_1d: the JAX (differentiable) Jones path does not yet "
            "differentiate duty_cycle -- gradients flow to the index / depth / "
            "wavelength / angle parameters, but the duty moving-mesh rebuild is "
            "implemented only on the scalar pmm_efficiency_1d path.  Pass a "
            "CONCRETE duty_cycle, or use pmm_efficiency_1d for a duty gradient.")
    if not (0.0 < duty_c < 1.0):
        raise ValueError(
            "pmm_jones_1d: the JAX (differentiable) path needs a strictly "
            f"interior duty_cycle (0 < duty < 1), got {duty_c}.")
    d_wall_c = duty_c * period_c
    static = _jpmm_build_static(period_c, d_wall_c, int(degree), 1, 1,
                                bool(grade))
    orders = _jpmm_order_set(static, period_c, wl_c, n_max,
                             int(far_field_orders), int(degree), "pmm_jones_1d")

    n_substrate = jnp.asarray(n_substrate, cj)
    n_superstrate = jnp.asarray(n_superstrate, cj)
    depth = jnp.asarray(depth)
    wavelength = jnp.asarray(wavelength)
    eps_sup = n_superstrate ** 2
    eps_sub = n_substrate ** 2

    # Oblique Bloch wavenumber kx0 = Re(n_sup) sin(angle) k0 -- TRACED so
    # d/d(angle) and d/d(n_superstrate) flow.  A concrete angle == 0 (and NOT
    # traced) passes the python literal 0.0 for the byte-equal normal-incidence
    # branch (no convection / kx0^2 / Ez normaliser shift).
    angle_traced = is_jax_array(angle)
    if angle_traced or float(angle) != 0.0:
        k0_kx = 2.0 * jnp.pi / wavelength
        kx0 = jnp.real(n_superstrate) * jnp.sin(jnp.asarray(angle)) * k0_kx
    else:
        kx0 = 0.0

    # duty held fixed on the Jones surface (no moving-mesh phase yet); the
    # projection is the static numpy one.
    Tp = jnp.asarray(_jpmm_fourier_projection(orders, period_c, static), cj)

    eig = _jax_eig_stable()
    o, R, T, J = _jpmm_jones_solve(static, orders, Tp, jnp, eig, period_c,
                                   t_ridge, t_groove, eps_sup, eps_sub, depth,
                                   wavelength, kx0=kx0)
    return jnp.asarray(orders), R, T, J



# ===========================================================================
# PMMStack -- multilayer 1-D PMM (the spectral-element analogue of RCWAStack)
# ===========================================================================
#
# Each layer is a 1-D piecewise-constant (anisotropic) profile; the layers are
# composed vertically by the same Redheffer scattering-matrix recursion the
# single-layer solver uses.  The one structural requirement is that every layer
# share ONE nodal grid so the interface mode-match is dimensionally compatible
# -- so the stack is solved on the UNION of every layer's walls (a wall lands on
# every element boundary, eps exact per element).  Anisotropic / Jones throughout
# (scalar layers are promoted to isotropic tensors); normal or oblique incidence.

def _tensor3_dict(M):
    """``(3,3)`` (or scalar) permittivity -> the dict of all 9 components the
    tensor spectral-element assembler consumes (the in-plane subset drives the
    vertical/slant path; the out-of-plane components are zero for an in-plane
    tensor and carry the full-3x3 anisotropy when present -- mirrors
    :func:`_t3_slant`)."""
    M = np.asarray(M, dtype=_C)
    if M.ndim == 0:
        M = M * np.eye(3, dtype=_C)
    return dict(exx=M[0, 0], exy=M[0, 1], exz=M[0, 2],
                eyx=M[1, 0], eyy=M[1, 1], eyz=M[1, 2],
                ezx=M[2, 0], ezy=M[2, 1], ezz=M[2, 2])



def _pmm_union_grid(layer_segments, min_feature=None):
    """Build the shared nodal grid for a stack: the union of every layer's walls.

    ``layer_segments[i]`` = layer ``i``'s ``[(width_fraction, eps), ...]``.
    Returns ``(union_widths, layer_eps_union)`` where ``union_widths`` are the
    fractional widths of the union cells (sum to 1) and ``layer_eps_union[i][c]``
    is layer ``i``'s permittivity on union cell ``c`` (each union cell lies wholly
    within one of layer ``i``'s segments, so eps is exact per cell).

    ``min_feature`` (FRACTION of the period; device-geometry roadmap item 3a,
    2026-06-10): PHYSICAL wall-snap.  Distinct layers' staircase walls collide
    at offsets far above float noise (a 2-deg taper at n_slices=8 puts walls
    ~1.2 nm apart -> near-zero-width union elements -> a passive-but-wrong or
    blowing-up solve).  Adjacent union walls closer than ``min_feature`` are
    snapped to their midpoint and a warning names the merged pairs -- but ONLY
    when the pair comes from DIFFERENT layers: a close pair owned by a single
    layer is that layer's own intentional thin feature (a 1 nm liner) and is
    never thinned."""
    walls = {0.0, 1.0}
    wall_owners = {0.0: set(), 1.0: set()}
    cums = []
    for li, segs in enumerate(layer_segments):
        w = np.asarray([float(s[0]) for s in segs], dtype=float)
        cw = np.concatenate([[0.0], np.cumsum(w)])
        cw[-1] = 1.0
        cums.append(cw)
        for x in cw:
            x = float(x)
            walls.add(x)
            wall_owners.setdefault(x, set()).add(li)
    uwalls = np.array(sorted(walls))
    # MERGE near-coincident walls (geometry-side conditioning fix): a tapered
    # stack offsets each slice's walls by ~dz*tan(theta), so the union of
    # different layers' walls contains pairs differing by floating noise / sub-pm
    # amounts -> sub-nm or zero-width union cells -> a J~0 spectral element -> a
    # singular S0 (equilibration alone cannot rescue a genuinely zero-width
    # element).  Snap walls closer than ``tol`` (fractional; ~sub-pm for any
    # realistic period) together; intentional thin features (a 1 nm liner is a
    # ~1e-3 fraction) are far above tol and untouched.  Per-cell eps is still
    # assigned by midpoint below, so the merge never mislabels a region.
    if uwalls.size > 2:
        tol = 1e-9
        keep = [uwalls[0]]
        owners = [wall_owners.get(float(uwalls[0]), set())]
        for w in uwalls[1:]:
            if w - keep[-1] > tol:
                keep.append(w)
                owners.append(wall_owners.get(float(w), set()))
            else:
                owners[-1] = owners[-1] | wall_owners.get(float(w), set())
        if keep[-1] < uwalls[-1]:           # never drop the period boundary
            keep[-1] = uwalls[-1]
        # ---- PHYSICAL wall-snap (item 3a): cross-layer pairs only ----------
        if min_feature is not None and float(min_feature) > tol:
            mf = float(min_feature)
            merged_pairs = []
            out_w = [keep[0]]
            out_o = [owners[0]]
            for w, ow in zip(keep[1:], owners[1:]):
                d = w - out_w[-1]
                interior = 0.0 < out_w[-1] and w < 1.0
                if (d < mf and interior
                        and not (out_o[-1] & ow)):      # no common owner layer
                    merged_pairs.append((out_w[-1], w))
                    out_w[-1] = 0.5 * (out_w[-1] + w)   # snap to midpoint
                    out_o[-1] = out_o[-1] | ow
                else:
                    out_w.append(w)
                    out_o.append(ow)
            if merged_pairs:
                import warnings as _warnings
                pairs_txt = ", ".join(
                    f"({a:.6g}, {b:.6g})" for a, b in merged_pairs[:6])
                more = ("" if len(merged_pairs) <= 6
                        else f" (+{len(merged_pairs) - 6} more)")
                _warnings.warn(
                    f"_pmm_union_grid: snapped {len(merged_pairs)} pair(s) of "
                    f"NEAR-COINCIDENT cross-layer walls closer than "
                    f"min_feature={mf:.3g} (period fractions): {pairs_txt}"
                    f"{more}.  These near-zero-width union elements are the "
                    f"staircase wall-collision pathology (passive-but-wrong / "
                    f"blow-up); the snap moves each pair to its midpoint.  "
                    f"Single-layer thin features are never merged.",
                    stacklevel=3)
            keep = out_w
        uwalls = np.array(keep)
    uwidths = np.diff(uwalls)
    mids = 0.5 * (uwalls[:-1] + uwalls[1:])
    layer_eps_union = []
    for segs, cw in zip(layer_segments, cums):
        row = []
        for m in mids:
            j = min(max(int(np.searchsorted(cw, m, side="right") - 1), 0),
                    len(segs) - 1)
            row.append(segs[j][1])
        layer_eps_union.append(row)
    return uwidths, layer_eps_union



# ===========================================================================
# Slanted 1-D lamellar grating -- inclined-coordinate PMM (Granet 2017)
# ===========================================================================
# A slanted grating has straight side-walls tilted by ``slant_angle`` from the
# vertical (z) axis.  A Fourier method (RCWA/FMM) cannot represent the slant
# directly -- it must STAIRCASE the layer into many thin laterally-shifted
# binary sub-layers and converge in the slice count.  In the inclined
# coordinate ``u = x - tan(phi) z`` the slant walls become coordinate surfaces
# ``u = const``, so ``eps`` depends on ``u`` only and a SINGLE slanted layer is
# solved exactly -- no z-staircase.  The slant injects a LINEAR-in-q convection
# term, making the modal eigenproblem QUADRATIC (companion-linearized); it also
# breaks the +/-q field symmetry (like a full-3x3 tensor layer), so the
# explicit forward/backward GENERALIZED S-matrix is used.  Validated vs an RCWA
# staircase: NORMAL incidence, slant 0-75deg, TE+TM, reaching the converged
# efficiencies at a single-layer DOF the staircase needs ~30-70x more work for.
def _build_sem_slant(period, d_wall, eps_ridge, eps_groove, degree,
                     n_ridge_el, n_groove_el, grade):
    """Assemble the periodic C0 spectral-element operators on the inclined
    coordinate ``u = x - tan(phi) z``.  Element boundaries land on the walls
    ``u = 0`` and ``u = d_wall``, so ``eps`` is constant per element and every
    Galerkin integral is exact.  Adds the convection operators ``C = INT B B'``
    and ``Cinv = INT (1/eps) B B'`` (the slant's linear-in-q term) to the usual
    mass/stiffness set."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    rb = _graded_boundaries(0.0, d_wall, n_ridge_el, grade)
    gb = _graded_boundaries(d_wall, period, n_groove_el, grade)
    elem_bnds = (list(zip(rb[:-1], rb[1:], [eps_ridge] * n_ridge_el))
                 + list(zip(gb[:-1], gb[1:], [eps_groove] * n_groove_el)))
    n_el = len(elem_bnds)

    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n_glob = last

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    S0, Peps, Pinv, L, Linv, C, Cinv = (_z() for _ in range(7))
    for e in range(n_el):
        xl, xr, eps = elem_bnds[e]
        J = 0.5 * (xr - xl)
        inv = 1.0 / eps
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Kloc = (Dphys.T * wel) @ Dphys          # INT phi' phi'  (stiffness)
        Cloc = Mloc @ Dphys                     # INT phi phi'   (convection)
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        S0[ix] += Mloc
        Peps[ix] += eps * Mloc
        Pinv[ix] += inv * Mloc
        L[ix] += Kloc
        Linv[ix] += inv * Kloc
        C[ix] += Cloc
        Cinv[ix] += inv * Cloc
    return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, C=C, Cinv=Cinv,
                n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds, degree=degree,
                ref_nodes=ref_nodes)



def _sem_modes_slant(mats, k0, polarization, slant_angle, kx0=0.0):
    """Periodic modal eigenproblem in the inclined coordinate.

    The slant injects a LINEAR-in-q convection term, so the modal problem is
    QUADRATIC ``(A1 + q Ac + q^2 A2) phi = 0``, solved by companion
    linearization.  At ``slant_angle == 0`` (and ``kx0 == 0``) the convection
    term vanishes and this reduces EXACTLY to the vertical PMM generalized
    eigenproblem with the symmetric +/-q field convention.

    Returns a mode dict (``symmetric`` flag + the forward/backward field blocks
    + eigenvalues + the ``1/eps`` multiply operator ``invop`` for TM)."""
    # Sign: u = x + tan(phi) z so the PMM shear matches the RCWA staircase's
    # +tan(phi)*z ridge shift (else the diffraction orders come out mirrored
    # m -> -m; identical physics, only the order labels flip).
    t = -np.tan(slant_angle)
    sec2 = 1.0 / np.cos(slant_angle) ** 2
    k02 = k0 * k0
    n = mats["S0"].shape[0]
    Imat = np.eye(n, dtype=_C)
    Z = np.zeros((n, n), dtype=_C)
    skip_slant = abs(t) < 1e-10                  # tiny-phi cancellation guard

    # nodal first-derivative operator Dop = S0^-1 C (L2-projected derivative);
    # used in the slant convection term and the slant-aware modal H-partner.
    Dop = _safe_solve(mats["S0"], mats["C"])

    if polarization == "te":
        S0, Peps = mats["S0"], mats["Peps"]
        Lop = mats["L"]
        Cv = mats["C"]
        if kx0:
            Cas = Cv - Cv.T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * S0
        # quadratic pencil (A1 - q Ac - q^2 A2) phi = 0:
        #   A1 = Peps - sec^2 Lop/k0^2,  Ac = (2 i t/k0) C,  A2 = S0.
        A1 = Peps - sec2 * Lop / k02
        A2 = S0
        Ac = (2j * t / k0) * Cv
        invop = None
    else:  # tm: built from 1/eps (Li inverse rule)
        S0, Pinv = mats["S0"], mats["Pinv"]
        Lop = mats["Linv"]
        Cv = mats["Cinv"]
        if kx0:
            Cas = Cv - Cv.T
            Lop = Lop - 1j * kx0 * Cas + (kx0 * kx0) * Pinv
        A1 = S0 - sec2 * Lop / k02
        A2 = Pinv
        # TM convection is the ANTISYMMETRIZED (1/eps)-weighted form
        #   Ac = (i t / k0)(Cinv - Cinv^T)
        # (the 1/eps sits BETWEEN the two z-derivatives in TM, so the slant
        # cross term integrates by parts to Cinv - Cinv^T, NOT 2 Cinv).
        Ac = (1j * t / k0) * (Cv - Cv.T)
        invop = _safe_solve(mats["S0"], mats["Pinv"])

    if skip_slant:
        # VERTICAL layer (any incidence): the +/-q field symmetry holds, so this
        # is the standard PMM generalized eig A1 phi = q^2 A2 phi (the kx0 Bloch
        # shift already lives in A1's Lop).  A2 = Pinv is the well-conditioned
        # mass, so fold to the faster standard eig (guarded; cf. _fast_geig).
        q2, Acoef = _fast_geig(A1, A2)
        q = np.sqrt(q2)
        # NOISE-ROBUST forward branch, unconditional (v5.14 P1 fix, the same
        # legacy ``Im(q) < 0`` flip as _sem_modes: it flipped propagating
        # modes on ~1e-15 QZ noise at normal incidence -- BLAS-build-dependent,
        # which broke the slant-zero == vertical reduction on CI builds where
        # the noise signs differ from the dev box).
        q = _forward_branch_flip(q)               # shared selector (S1-8)
        lam = -1j * q
        return dict(symmetric=True, W=Acoef, q=q, lam=lam, invop=invop,
                    Dop=Dop, t=t, k0=k0, polarization=polarization)

    # ---- general case: linearize the quadratic pencil to a 2n first-order
    # generalized eigenproblem and pick the FORWARD half by z-Poynting flux ----
    Abig = np.block([[Z, Imat], [A1, -Ac]])
    Bbig = np.block([[Imat, Z], [Z, A2]])
    qall, Vbig = sla.eig(Abig, Bbig)
    finite = np.isfinite(qall)
    qall = qall[finite]
    phis = Vbig[:n, finite]                       # upper block = phi
    nrm = np.linalg.norm(phis, axis=0)
    nrm = np.where(nrm < 1e-300, 1.0, nrm)
    phis = phis / nrm
    q = qall

    # slant-aware modal tangential partner (continuous across the PLANAR
    # z-interfaces), from the lab-frame d/dz = (i gamma - t d/du):
    #   TE: V = q W + (i t/k0) Dop W
    #   TM: V = q (1/eps) W + (i t/k0)(1/eps) Dop W  (derivative BEFORE 1/eps)
    if polarization == "te":
        V = phis * q[None, :] + (1j * t / k0) * (Dop @ phis)
    else:
        V = (invop @ phis) * q[None, :] \
            + (1j * t / k0) * (invop @ (Dop @ phis))

    # forward selector by z-Poynting flux Sz ~ Re(F conj(Hx)); propagating modes
    # keep Sz > 0, evanescent keep Im(q) > 0 (decay exp(+i q k0 z)).
    SF = S0 @ np.conj(V)
    Sz = np.real(np.einsum("in,in->n", phis, SF))
    qmax = max(float(np.max(np.abs(q))), 1.0)
    tol = 1e-7 * qmax
    prop = np.abs(q.imag) < tol
    fwd = np.where(prop, Sz > 0.0, q.imag > 0.0)
    fidx = np.where(fwd)[0]
    if fidx.size != n:                            # rebalance to exactly n forward
        # Sz (z-flux, ~length^2) and q.imag (decay, dimensionless) are NOT on a
        # common scale, so a single argsort over a mixed score can rank a strongly
        # decaying evanescent mode above a weakly-propagating one (audit F5).
        # Rank each pool on its OWN measure and PREFER propagating-forward modes
        # (net forward power) over evanescent (forward-decaying) ones.  In
        # practice the re-ranked modes near the boundary are the flux-null
        # spurious sea (zero power), so observables are unchanged -- this only
        # removes the unit-mixing fragility.
        prop_i = np.where(prop)[0]
        ev_i = np.where(~prop)[0]
        prop_i = prop_i[np.argsort(-Sz[prop_i])]
        ev_i = ev_i[np.argsort(-q.imag[ev_i])]
        fidx = np.sort(np.concatenate([prop_i, ev_i])[:n])
    bidx = np.array(sorted(set(range(len(q))) - set(fidx.tolist())))

    Wf, Vf, qf = phis[:, fidx], V[:, fidx], q[fidx]
    Wb, Vb, qb = phis[:, bidx], V[:, bidx], q[bidx]
    lam_f = -1j * qf                              # exp(-lam_f k0 z) decays fwd
    lam_b = -1j * qb
    return dict(symmetric=False, Wf=Wf, Vf=Vf, lam_f=lam_f, qf=qf,
                Wb=Wb, Vb=Vb, lam_b=lam_b, qb=qb, invop=invop,
                Dop=Dop, t=t, k0=k0, polarization=polarization)



def _sem_modes_slant_uniform(mats, k0, polarization, eps, kx0=0.0, geo=None,
                             Dop=None):
    """Homogeneous half-space slant-mode dict from the SHARED eps-free geometric
    eig (audit S5-P1 for the slant path).

    A uniform medium has no walls -> ``slant_angle = 0`` (``t = 0``), so the
    ``+/-q`` symmetric convention holds and BOTH the TE ``(eps S0 - Lop/k0^2)x =
    q^2 S0 x`` and TM ``(S0 - Linv/k0^2)x = q^2 Pinv x`` problems fold (``Linv =
    L/eps``, ``Pinv = S0/eps`` for uniform eps) to the SAME eps-FREE pencil
    ``(Lop/k0^2)x = (eps - q^2)S0 x`` -- exactly :func:`_scalar_uniform_geo_eig`.
    So ONE eig serves the superstrate AND substrate half-spaces (shared mesh ->
    identical ``L/C/S0``) AND both polarizations, with ``q^2 = eps - mu`` and the
    IDENTICAL forward-branch flip as the ``skip_slant`` branch of
    :func:`_sem_modes_slant`.  ``invop = (1/eps) I`` exactly (``Pinv = (1/eps)S0``
    for a uniform medium, so ``S0^-1 Pinv = (1/eps)I`` -- no solve).  The
    eigenvector gauge / order may differ from a raw per-eps eig, but the
    downstream interface S-matrix + far-field lstsq are gauge-agnostic (the same
    guarantee S5-P1 uses for the vertical half-spaces)."""
    if geo is None:
        geo = _scalar_uniform_geo_eig(mats, k0, kx0)
    mu, X = geo
    q = np.sqrt(_C(eps) - mu)
    q = _forward_branch_flip(q)                   # shared selector (S1-8)
    lam = -1j * q
    n = mats["S0"].shape[0]
    invop = None if polarization == "te" else np.eye(n, dtype=_C) / _C(eps)
    if Dop is None:
        Dop = _safe_solve(mats["S0"], mats["C"])
    return dict(symmetric=True, W=X, q=q, lam=lam, invop=invop,
                Dop=Dop, t=-0.0, k0=k0, polarization=polarization)


def _modes_M_slant(md):
    """Build the generalized field-mode matrix ``M = [[Wf, Wb], [Vf, Vb]]`` plus
    ``(lam_f, lam_b)`` and the forward field block ``Wf`` from a
    :func:`_sem_modes_slant` dict.  For the symmetric (vertical) case the
    backward modes are the mirror ``[W; -V]`` with eigenvalue ``-lam`` -- the
    implicit convention of the symmetric interface -- so the same generalized
    recursion serves both cases."""
    pol = md["polarization"]
    if md["symmetric"]:
        W, q, lam = md["W"], md["q"], md["lam"]
        invop = md["invop"]
        Wbase = W if pol == "te" else (invop @ W)
        Vf = Wbase * q[None, :]                  # forward partner (+q)
        Wf, lam_f = W, lam
        Wb, Vb, lam_b = W, -Vf, -lam             # mirror backward
        M = np.block([[Wf, Wb], [Vf, Vb]])
        return M, lam_f, lam_b, Wf
    Wf, Vf, lam_f = md["Wf"], md["Vf"], md["lam_f"]
    Wb, Vb, lam_b = md["Wb"], md["Vb"], md["lam_b"]
    M = np.block([[Wf, Wb], [Vf, Vb]])
    return M, lam_f, lam_b, Wf



def _pmm_slant_solve(period, n_ridge, n_groove, n_substrate, n_superstrate,
                     depth, duty_cycle, wavelength, slant_angle, *, angle,
                     polarization, degree, elements_per_region, grade,
                     far_field_orders, return_coeffs=False):
    """Single-layer slanted-grating PMM solve (the inclined-coordinate layer +
    homogeneous half-spaces + generalized S-matrix + lab-frame far field).
    Returns ``(orders, R, T, n_glob)``.

    With ``return_coeffs=True`` ALSO returns the COMPLEX zeroth-order reflected /
    transmitted field coefficients ``(r0, t0)`` -> ``(orders, R, T, n_glob, r0,
    t0)``.  The default ``return_coeffs=False`` is byte-identical to the legacy
    return (the efficiency caller :func:`pmm_efficiency_1d_slanted` is
    untouched); the coefficient tail feeds the DIAGONAL-tensor cure in
    :func:`pmm_jones_1d_slanted` (assembling the diagonal Jones from the two
    div-conforming scalar channels)."""
    eps_ridge, eps_groove = _C(n_ridge) ** 2, _C(n_groove) ** 2
    eps_sup, eps_sub = _C(n_superstrate) ** 2, _C(n_substrate) ** 2
    k0 = 2.0 * np.pi / wavelength
    d_wall = duty_cycle * period
    kx0 = float(np.real(n_superstrate)) * np.sin(float(angle)) * k0
    nel = int(elements_per_region)

    mats = _build_sem_slant(period, d_wall, eps_ridge, eps_groove, degree,
                            nel, nel, grade)
    # The half-spaces are HOMOGENEOUS -> no slant (a uniform medium has no
    # walls); build them with slant_angle = 0 (their modes are plane waves).
    mats_sup = _build_sem_slant(period, d_wall, eps_sup, eps_sup, degree,
                                nel, nel, grade)
    mats_sub = _build_sem_slant(period, d_wall, eps_sub, eps_sub, degree,
                                nel, nel, grade)
    n_max = max(np.real(n_superstrate), np.real(n_substrate), np.real(n_ridge),
                np.real(n_groove))
    n_glob = mats["n_glob"]

    m_prop = _n_propagating_orders(period, wavelength, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"pmm_efficiency_1d_slanted: degree={degree} too low to resolve "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}).")
    orders = np.arange(-half, half + 1)
    G = 2.0 * np.pi / period
    kx = (kx0 + orders * G) / k0
    Tp = _sem_fourier_projection(orders, period, mats)

    # layer modes (inclined) + homogeneous half-space modes (lab/vertical), each
    # assembled into the explicit forward/backward field-mode matrix for the
    # GENERALIZED S-matrix (the slant breaks the +/-q symmetry).
    md_l = _sem_modes_slant(mats, k0, polarization, slant_angle, kx0)
    # S5-P1 (slant): the two HOMOGENEOUS half-spaces share the mesh (identical
    # L/C/S0) and are slant-free, so ONE eps-free geometric eig (+ one shared
    # Dop) serves both -- q^2 = eps - mu, invop = (1/eps)I.  Gauge-equivalent to
    # the per-eps _sem_modes_slant eig (interface S-matrix is gauge-agnostic).
    _geo_hs = _scalar_uniform_geo_eig(mats_sup, k0, kx0)
    _Dop_hs = _safe_solve(mats_sup["S0"], mats_sup["C"])
    md_s = _sem_modes_slant_uniform(mats_sup, k0, polarization, eps_sup, kx0,
                                    _geo_hs, _Dop_hs)
    md_b = _sem_modes_slant_uniform(mats_sub, k0, polarization, eps_sub, kx0,
                                    _geo_hs, _Dop_hs)
    Ml, lamf_l, lamb_l, _Wf_l = _modes_M_slant(md_l)
    Ms, _lamf_s, _lamb_s, Wf_s = _modes_M_slant(md_s)
    Mb, _lamf_b, _lamb_b, Wf_b = _modes_M_slant(md_b)

    S = _interface_smatrix_general(Ms, Ml)
    S = _propagation_star_general(S, lamf_l, lamb_l, k0 * depth)
    S = _redheffer_star(S, _interface_smatrix_general(Ml, Mb))
    S11, _S12, S21, _S22 = S

    # forward far-field projection: the forward field block of each (homogeneous)
    # half-space is a set of plane waves; project order-0 incidence, recover r/t.
    Hsup = Tp @ Wf_s
    Hsub = Tp @ Wf_b
    delta0 = (orders == 0).astype(_C)
    cinc, *_ = np.linalg.lstsq(Hsup, delta0, rcond=None)
    r_ord = Hsup @ (S11 @ cinc)
    t_ord = Hsub @ (S21 @ cinc)

    R, T = _scalar_farfield_RT(r_ord, t_ord, kx, kx0, k0, eps_sup, eps_sub,
                               polarization, label="pmm_efficiency_1d_slanted")
    if return_coeffs:
        m0 = int(np.where(orders == 0)[0][0])
        return orders, R, T, n_glob, _C(r_ord[m0]), _C(t_ord[m0])
    return orders, R, T, n_glob



# ===========================================================================
# SLANTED + ANISOTROPIC (Jones) PMM -- the CONVECTION METRIC GENERATOR
# (Edee-Granet 2024 [E_t; H_t] first-order metric generator; slant carried as
# tan*d/dx convection).
# ---------------------------------------------------------------------------
# NAMING: this is the "convection" slant path (factorization='convection') -- the
# lab-Cartesian metric generator.  It was historically called the "covariant-
# metric" generator; that name is RETIRED here to avoid colliding with the
# genuinely-covariant Li-1999 oblique-coordinate path (factorization='covariant',
# the _cov_* family below), which is a DIFFERENT operator (spectral vs algebraic).
# Slant enters ONLY through the metric-folded effective tensors eps^lm / mu^lm;
# the eigenproblem is the TRUE first-order physical Maxwell operator
#     -i k gamma psi = L psi,   L = A + B C^-1 D,   psi = [Ex; Ey; iZ Hx; iZ Hy]
# (Edee & Granet 2024, JOSA A 41(9) 1803, Eqs. 3, 7-15), LINEAR in gamma.  H is a
# genuine STATE component read directly from the eigenvector (the lower G-block =
# iZ H), so the modes are energy-conserving BY CONSTRUCTION -- the property a
# reshaped convection pencil lacks.  Validated (proto round 11): energy ~1e-13 all
# slants 0-60 BOTH real-symmetric AND gyrotropic, reduces to pmm_jones_1d at phi=0
# and to pmm_efficiency_1d_slanted on the diagonal, linear-in-gamma residual
# ~6e-16.  The round-19 div-conforming E_z closure (1/ezz BETWEEN the discrete
# z-derivatives) + kx0 Bloch wiring extend it to COMBINED OBLIQUE + SLANT (the
# generator + lab half-spaces conserve + are per-order accurate degree-cleanly).
# SCOPE: binary (1 ridge + 1 groove), normal OR oblique incidence at any slant.
# ===========================================================================

def _t3_slant(M):
    """Full (3, 3) tensor -> the dict of all 9 components the metric generator's
    nodal builder + the full-3x3 out-of-plane generator consume.  The in-plane
    subset (``exx, exy, eyx, eyy, ezz``) drives the in-plane / slant path; the
    out-of-plane components (``exz, eyz, ezx, ezy``) are zero for an in-plane
    tensor (so the generator's ``off_present`` flag stays False and the operator
    is byte-identical to the in-plane path) and carry the full-3x3 anisotropy when
    present."""
    return dict(exx=M[0, 0], exy=M[0, 1], exz=M[0, 2],
                eyx=M[1, 0], eyy=M[1, 1], eyz=M[1, 2],
                ezx=M[2, 0], ezy=M[2, 1], ezz=M[2, 2])



def _build_nodal_metric(period, d_wall, t_ridge, t_groove, degree,
                        n_ridge_el, n_groove_el, grade):
    """Nodal Galerkin operator builder for the metric generator (mirrors the
    :func:`_build_sem_tensor` element loop).

    Returns the unit mass ``S0``, the per-coefficient masses (DIRECT and
    ``1/exx`` / ``1/ezz`` INVERSE already separated), and the convection
    ``C = INT phi phi'``.  All metric folding / inverse-rule logic lives in
    :func:`_build_generator_metric`."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    rb = _graded_boundaries(0.0, d_wall, n_ridge_el, grade)
    gb = _graded_boundaries(d_wall, period, n_groove_el, grade)
    elem_bnds = (list(zip(rb[:-1], rb[1:], [t_ridge] * n_ridge_el))
                 + list(zip(gb[:-1], gb[1:], [t_groove] * n_groove_el)))
    n_el = len(elem_bnds)
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n_glob = last

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    # masses keyed by the coefficient they integrate
    keys = ("one", "exx", "eyy", "exy", "eyx", "ezz", "inv_exx", "inv_ezz")
    mass = {k: _z() for k in keys}
    C = _z()
    # DIV-CONFORMING longitudinal operators for the Gauss-law-clean Ez closure
    # (round 19): 1/ezz placed BETWEEN the two z-derivatives -- the placement the
    # div-conforming Ez-elimination in :func:`_build_generator_metric` needs
    # (Granet 2023 Eq.16-18 / Popov-Neviere App.B), vs the spurious-prone pointwise
    # [[1/ezz]] average (= ``mass['inv_ezz']``).  ``L_inv_ezz = INT(1/ezz) B' B'``
    # is the 1/ezz-weighted z-STIFFNESS; ``C_inv_ezz = INT(1/ezz) B B'`` the
    # 1/ezz-weighted convection (for the kx0 Bloch antisymmetrization at oblique
    # incidence).  Cheap to accumulate here; inert unless the generator uses them.
    L_inv_ezz = _z()
    C_inv_ezz = _z()
    for e in range(n_el):
        xl, xr, t = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Cloc = Mloc @ Dphys
        Kloc = (Dphys.T * wel) @ Dphys           # INT B' B' (element stiffness)
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        mass["one"][ix] += Mloc
        mass["exx"][ix] += exx * Mloc
        mass["eyy"][ix] += eyy * Mloc
        mass["exy"][ix] += exy * Mloc
        mass["eyx"][ix] += eyx * Mloc
        mass["ezz"][ix] += ezz * Mloc
        mass["inv_exx"][ix] += (1.0 / exx) * Mloc
        mass["inv_ezz"][ix] += (1.0 / ezz) * Mloc
        L_inv_ezz[ix] += (1.0 / ezz) * Kloc
        C_inv_ezz[ix] += (1.0 / ezz) * Cloc
        C[ix] += Cloc
    return dict(mass=mass, C=C, S0=mass["one"], n_glob=n_glob, l2g=l2g,
                elem_bnds=elem_bnds, degree=degree, ref_nodes=ref_nodes,
                L_inv_ezz=L_inv_ezz, C_inv_ezz=C_inv_ezz)



def _build_nodal_metric_segments(period, widths, seg_tensors, degree,
                                 n_el_per_region, grade):
    """N-region generalization of :func:`_build_nodal_metric` for the metric
    generator.  ``seg_tensors[i] = dict(exx, exy, eyx, eyy, ezz)`` of region
    ``i``; ``widths[i]`` its fractional width.  Mirrors the binary element loop
    EXACTLY -- only the element table comes from :func:`_segment_elem_bnds`
    (N graded regions, a wall on every boundary, the FMM-matching reversed
    layout) instead of the ridge|groove split.  Returns the same dict shape so
    :func:`_build_generator_metric` / :func:`_layer_modes_metric` consume it
    unchanged (the operator + far field are region-count-agnostic)."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    elem_bnds = _segment_elem_bnds(period, widths, seg_tensors, n_el_per_region,
                                   grade)
    n_el = len(elem_bnds)
    l2g, n_glob = _l2g_periodic(n_el, degree)

    def _z():
        return np.zeros((n_glob, n_glob), dtype=_C)
    keys = ("one", "exx", "eyy", "exy", "eyx", "ezz", "inv_exx", "inv_ezz")
    mass = {k: _z() for k in keys}
    C = _z()
    L_inv_ezz = _z()
    C_inv_ezz = _z()
    for e in range(n_el):
        xl, xr, t = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Cloc = Mloc @ Dphys
        Kloc = (Dphys.T * wel) @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        mass["one"][ix] += Mloc
        mass["exx"][ix] += exx * Mloc
        mass["eyy"][ix] += eyy * Mloc
        mass["exy"][ix] += exy * Mloc
        mass["eyx"][ix] += eyx * Mloc
        mass["ezz"][ix] += ezz * Mloc
        mass["inv_exx"][ix] += (1.0 / exx) * Mloc
        mass["inv_ezz"][ix] += (1.0 / ezz) * Mloc
        L_inv_ezz[ix] += (1.0 / ezz) * Kloc
        C_inv_ezz[ix] += (1.0 / ezz) * Cloc
        C[ix] += Cloc
    return dict(mass=mass, C=C, S0=mass["one"], n_glob=n_glob, l2g=l2g,
                elem_bnds=elem_bnds, degree=degree, ref_nodes=ref_nodes,
                L_inv_ezz=L_inv_ezz, C_inv_ezz=C_inv_ezz)



def _coeff_mass_metric(mats, fn):
    """Assemble the Galerkin mass ``INT phi fn(material) phi`` for an arbitrary
    scalar coefficient ``fn(t_dict)`` (element-piecewise constant), reusing the
    element table.  Returns the ``n x n`` mass (NOT yet ``iS0``-applied)."""
    ref_nodes, ref_w = _gll_nodes_weights(mats["degree"])
    elem_bnds = mats["elem_bnds"]
    l2g = mats["l2g"]
    n = mats["n_glob"]
    M = np.zeros((n, n), dtype=_C)
    for e in range(len(elem_bnds)):
        xl, xr, t = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Mloc = np.diag(wel)
        idx = l2g[e]
        M[np.ix_(idx, idx)] += fn(t) * Mloc
    return M



def _build_generator_metric(mats, k0, slant_angle, kx0=0.0):
    r"""Assemble the ``4n x 4n`` physical first-order Maxwell generator ``L`` for
    the state ``psi = [Ex; Ey; iZ Hx; iZ Hy]``.  Slant enters ONLY through the
    metric-folded ``eps^lm`` / ``mu^lm``.  Returns ``(L, n)``.

    LOAD-BEARING metric fold (validated; do NOT "fix"): the contravariant
    ``eps^lm = sqrt(g) [J^-1 eps_mat J^-T]^lm`` (the in-plane-anisotropic
    realization, with ``J = [[1,0,tan],[0,1,0],[0,0,1]]``, ``sqrt(g)=1``):
        eps^11 = exx + ezz tan^2   (wall-normal: inverse rule on 1/(exx+ezz t^2))
        eps^13 = eps^31 = -ezz tan ;  eps^33 = ezz ;  eps^22 = eyy
        eps^12 = exy ; eps^21 = eyx ; eps^23 = eps^32 = 0
        mu^lm = sqrt(g) g^lm:  mu^11 = sec2 ; mu^13 = mu^31 = -tan ; mu^33 = 1.
    ``eps^13 = -ezz tan`` and ``mu^13 = -tan`` share the sign of ``tan`` -> TE
    and TM convect identically (the metric is self-consistent).  Each block
    coefficient is a nodal operator (``d1 -> Dop = S0^-1 C``; ``eps^lm/mu^lm ->
    iS0 @ Galerkin mass``), with the Li INVERSE rule on the WALL-NORMAL component
    (``eps^11``) and DIRECT rule on wall-tangential (``eyy``, ``mu``).

    The longitudinal ``E_z`` closure (the ``(0,3)`` slot) is DIV-CONFORMING (round
    19): after ``L = A + B Cinv D`` the pointwise ``-(1/k) Dopx [[1/ezz]] Dopx`` is
    surgically replaced by the Li-inverse-rule ``+(1/k) iS0 INT(1/ezz) B' B'``
    (``1/ezz`` BETWEEN the z-derivatives) -- see the inline note.  ``kx0`` is the
    oblique Bloch shift (``Dop -> Dopx = Dop + i kx0`` in B/D, plus the kx0
    antisym-convection + mass in the ``1/ezz`` bracket); ``kx0 = 0`` is
    byte-identical to the normal-incidence generator for the in-plane blocks.
    """
    n = mats["n_glob"]
    k = k0                                   # k = omega sqrt(eps0 mu0) = k0
    S0 = mats["S0"]
    iS0 = _safe_inv(S0)
    mass = mats["mass"]
    I = np.eye(n, dtype=_C)
    Z = np.zeros((n, n), dtype=_C)

    # SLANT as EXACT CONVECTION (2026-06-07).  The slant is carried as a first-
    # order convection term `tan * d/dx` on each field block (added at the very
    # end of this function) rather than folded into a static `ezz*tan^2` wall-
    # normal term.  The static fold is the "slant cancels in the ezz-Schur, then
    # is re-injected as ezz*tan^2" approximation, which caps per-order accuracy at
    # ~1e-2 for strongly-coupled / steep-slant cells; the exact convection on the
    # CLEAN slant=0 base reaches the ~1e-4 wall-normal floor UNIFORMLY (validated
    # per-order vs a converged staircase-RCWA oracle: slant 15-60 deg, in-plane /
    # out-of-plane / lossy / asymmetric tensors, normal AND oblique; energy
    # conserves ~1e-13; the mild advection spurious it adds -- |Re q|~14 at slant
    # 45 vs a from-scratch convection form's ~210 -- are flux-null and absorbed by
    # the existing forward/backward selector).  So the eps^lm / mu^lm operators
    # below are built at slant=0 and the slant enters ONLY via the convection.
    tan_conv = np.tan(slant_angle)           # the slant, carried as convection
    # (material / mu operators are built at slant=0; the slant enters ONLY via the
    # tan_conv*Dopx convection added at the end -- NOT the archived ezz*tan^2 fold.)

    # ---- nodal derivative d1 (= D1) ----------------------------------------
    Dop = _safe_solve(S0, mats["C"])         # S0^-1 INT phi phi'
    # OBLIQUE incidence: the transverse derivative acting on the periodic envelope
    # is d/dx + i*kx0 (field ~ e^{i kx0 x} u(x), Granet Eq.17 d1 -> i alpha0 + d1).
    # Inject it into every transverse-derivative slot of B/D.  At kx0 = 0 (normal
    # incidence) Dopx == Dop, so the generator is byte-identical to the prior path.
    Dopx = Dop + 1j * kx0 * I

    # ---- FULL-3x3 OUT-OF-PLANE detection (the native off-plane extension) ----
    # The full-3x3 anisotropic case (exz/eyz/ezx/ezy != 0) is solved in THIS metric
    # A+B C^-1 D layout: the out-of-plane coupling enters via (a) the POINTWISE
    # ezz-Schur composites of the wall-normal / in-plane operators (Li 1999 Eq.12,
    # below) and (b) the surgical single-derivative cross blocks (after L is
    # assembled).  Both VANISH identically when exz=eyz=ezx=ezy=0, so off_present
    # False reproduces the in-plane / slant operator BYTE-FOR-BYTE.  (A naive
    # spectral B C^-1 D Schur of the RAW eps^13/31 -- forming the composites as
    # products of separately-discretized Toeplitz factors -- is the WRONG Li
    # factorization order and gives spurious modes; the pointwise composite is
    # correct.  V = -G is unchanged.)
    def _maxoff(key):
        return max(abs(complex(t[key])) for (_xl, _xr, t) in mats["elem_bnds"])
    off_present = any(_maxoff(kk) > 0.0 for kk in ("exz", "eyz", "ezx", "ezy"))

    Oezz = iS0 @ mass["ezz"]                  # [[ezz]]
    # (eps^33)^-1 = DIRECT average of 1/ezz (the wall-tangential longitudinal
    # inverse used inside the C^-1 elimination).
    O_inv_ezz = iS0 @ mass["inv_ezz"]         # [[1/ezz]]  (= (eps^33)^-1)
    if not off_present:
        # ---- IN-PLANE / SLANT material operators (byte-identical) ------------
        Oeyy = iS0 @ mass["eyy"]                 # [[eyy]]
        Oexy = iS0 @ mass["exy"]                 # [[exy]]
        Oeyx = iS0 @ mass["eyx"]                 # [[eyx]]
        # WALL-NORMAL inverse rule: [[1/exx]]^-1 (the discontinuous normal-D mult).
        # Slant enters as CONVECTION (tan_conv*Dopx, end of function), NOT a metric
        # fold, so the wall-normal / cross blocks are built at slant=0.  (The old
        # ezz*tan^2 fold that lived here is archived -- see _ARCHIVE_SLANT_FOLD.)
        Oinv_exx = iS0 @ mass["inv_exx"]         # [[1/exx]]
        Exx_norm = _safe_inv(Oinv_exx)           # [[1/exx]]^-1  (Li inverse rule)
        Oeps11 = Exx_norm                                     # = [[1/exx]]^-1
        Oeps13 = Z.copy()
        Oeps31 = Z.copy()
        Oeps22 = Oeyy
        Oeps12 = Oexy
        Oeps21 = Oeyx
    else:
        # ---- FULL-3x3 pointwise ezz-Schur composites (Li 1999 Eq.12) ---------
        # a_eff = exx - exz ezx/ezz, b_eff = exy - exz ezy/ezz,
        # c_eff = eyx - eyz ezx/ezz, d_eff = eyy - eyz ezy/ezz, formed element-
        # pointwise in x BEFORE the wall-normal inverse rule.  At off-plane=0 these
        # reduce EXACTLY to exx/exy/eyx/eyy.
        def _aeff(t_): return t_["exx"] - t_["exz"] * t_["ezx"] / t_["ezz"]
        def _beff(t_): return t_["exy"] - t_["exz"] * t_["ezy"] / t_["ezz"]
        def _ceff(t_): return t_["eyx"] - t_["eyz"] * t_["ezx"] / t_["ezz"]
        def _deff(t_): return t_["eyy"] - t_["eyz"] * t_["ezy"] / t_["ezz"]
        Oeps11 = _safe_inv(iS0 @ _coeff_mass_metric(
            mats, lambda t_: 1.0 / _aeff(t_)))   # slant via convection, not fold
        T_b_a = iS0 @ _coeff_mass_metric(mats, lambda t_: _beff(t_) / _aeff(t_))
        T_c_a = iS0 @ _coeff_mass_metric(mats, lambda t_: _ceff(t_) / _aeff(t_))
        T_sch = iS0 @ _coeff_mass_metric(
            mats, lambda t_: _deff(t_) - _ceff(t_) * _beff(t_) / _aeff(t_))
        Oeps12 = Oeps11 @ T_b_a
        Oeps21 = T_c_a @ Oeps11
        Oeps22 = T_sch + T_c_a @ Oeps11 @ T_b_a
        Oeps13 = Z.copy()                        # slant via convection, not fold
        Oeps31 = Z.copy()
    # mu (smooth metric, slant via convection -> identity; the ezz*tan^2 / -tan
    # metric fold that scaled these is archived as _ARCHIVE_SLANT_FOLD):
    Mu11 = I
    Mu13 = Z
    Mu31 = Z
    Mu33 = I
    Mu22 = I
    Mu23 = Z
    Mu32 = Z
    Mu12 = Z

    # ===== A (Eq.8) ; mu^21 = mu^12 = 0 here (slant couples 1-3, not 1-2) =====
    A = np.block([
        [Z,                Z,                (-k) * Mu12,    (-k) * Mu22],
        [Z,                Z,                (-k) * (-Mu11), (-k) * (-Mu12)],
        [(-k) * Oeps21,    (-k) * Oeps22,    Z,             Z],
        [(-k) * (-Oeps11), (-k) * (-Oeps12), Z,             Z],
    ])

    # ===== B (Eq.9, d2=0) (4x2) ; eps^23 = 0 =================================
    B = np.block([
        [Dopx,           (-k) * Mu23],
        [Z,              k * Mu13],
        [(-k) * Z,       Dopx],
        [k * Oeps13,     Z],
    ])

    # ===== C (Eq.11) ; C^-1 = -(1/k)[[0,(eps^33)^-1],[(mu^33)^-1,0]] =========
    #   (eps^33)^-1 = O_inv_ezz (DIRECT 1/ezz) ;  (mu^33)^-1 = I  (mu^33 = 1)
    Cinv = np.block([
        [Z,                 (-1.0 / k) * O_inv_ezz],
        [(-1.0 / k) * Mu33, Z],
    ])

    # ===== D (Eq.12, d2=0) (2x4) ; eps^32 = 0 ===============================
    D = np.block([
        [Z,           Dopx,     k * Mu31,  k * Mu32],
        [k * Oeps31,  (k) * Z,  Z,         Dopx],
    ])

    L = A + B @ Cinv @ D

    # ===== DIV-CONFORMING Ez closure (round 19; un-gated, all slants) ========
    # The pointwise [[1/ezz]] longitudinal closure that ``B @ Cinv @ D`` injects
    # into the (Ex-eqn, reads iZHy) slot (0,3) is
    #     L03_pointwise = -(1/k) Dopx [[1/ezz]] Dopx          (+stiffness, 1/ezz
    # OUTSIDE the z-derivatives) -- the discrete Gauss-law violation that admits
    # the Liu-2015 harmonic-mean spurious static null (amplified to 13-33% error
    # by the kx0 Bloch shift at oblique+slant).  Replace that single slot by the
    # Li-inverse-rule -stiffness  +(1/k) iS0 INT(1/ezz) B' B'  (1/ezz BETWEEN the
    # derivatives -- Granet 2023 Eq.16-18 / Popov-Neviere App.B), the SAME
    # placement the working scalar TM solver (:func:`_sem_modes_slant`) uses.  The
    # TM-block spectrum then bit-matches the scalar div-conforming TM solver and
    # the spurious null is gone (per-order TM improves ~9x coupled / ~300x
    # diagonal-exx!=ezz; energy still ~1e-13).  kx0 enters the longitudinal bracket
    # via Granet Eq.17: INT(1/ezz)(kx0 + d1)'(kx0 + d1) = L + kx0 antisym-conv +
    # kx0^2 mass.  The [E;H] block layout is untouched, so the magnetic partner
    # V = -G stays the genuine (energy-conserving) lower block.
    Lez = mats["L_inv_ezz"].copy()
    if kx0:
        Cas = mats["C_inv_ezz"] - mats["C_inv_ezz"].T
        Lez = Lez - 1j * kx0 * Cas + (kx0 * kx0) * mass["inv_ezz"]
    pointwise_long = (-1.0 / k) * (Dopx @ (O_inv_ezz @ Dopx))
    divconf_long = (1.0 / k) * (iS0 @ Lez)
    L[0:n, 3 * n:4 * n] += (divconf_long - pointwise_long)

    # ===== FULL-3x3 OUT-OF-PLANE single-derivative cross blocks ===============
    # The off-plane material coupling (exz/eyz/ezx/ezy) drives Ez and Hz through
    # (eps^33)^-1; eliminating them adds single-derivative terms into the two
    # quadrants the in-plane metric generator leaves ZERO: the A-quadrant (Ex/Ey
    # equations, rows 0..n) and the B-quadrant (iZHx/iZHy equations, rows 2n..4n).
    # Each carries the Li inverse rule inv([[ezz]]) (NOT the Laurent [[1/ezz]]) and
    # vanishes identically at off-plane=0.  Kx = Dopx/(i k0); scaled by k0 to the
    # metric L = k0 * G convention.
    if off_present:
        EZZi = _safe_inv(Oezz)                               # inv([[ezz]])
        EZX_L = iS0 @ _coeff_mass_metric(mats, lambda t_: t_["ezx"])
        EZY_L = iS0 @ _coeff_mass_metric(mats, lambda t_: t_["ezy"])
        EXZ_L = iS0 @ _coeff_mass_metric(mats, lambda t_: t_["exz"])
        EYZ_L = iS0 @ _coeff_mass_metric(mats, lambda t_: t_["eyz"])
        Kx = Dopx / (1j * k0)
        # FACTOR-i FIX (AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14, lockstep
        # with rcwa._core._layer_eigenmodes_tensor): the off-plane cross
        # blocks carry relative +/-i factors in the [E; iZH] metric state
        # (here A_new = -i * A_legacy on the E-rows, B_new = +i * B_legacy
        # on the iZH-rows -- the mirror of the rcwa assignment, matching
        # this generator's state/eigen conventions; pinned empirically
        # against the dispersion-anchored rcwa OOP solver, agreement
        # restored to the historical 1.5e-3 bar).  The legacy real
        # coefficients shared the rcwa generator's defect -- they agreed
        # with the PRE-fix rcwa OOP results for the same reason the
        # circular oracle did -- and gave the same artificially
        # +/- symmetric extraordinary dispersion.
        L[0:n, 0:n] += k0 * (1j * (Kx @ EZZi @ EZX_L))
        L[0:n, n:2 * n] += k0 * (1j * (Kx @ EZZi @ EZY_L))
        L[2 * n:3 * n, 3 * n:4 * n] += k0 * (-1j * (EYZ_L @ EZZi @ Kx))
        L[3 * n:4 * n, 3 * n:4 * n] += k0 * (1j * (EXZ_L @ EZZi @ Kx))

    # ===== SLANT as exact first-order convection (see top-of-function note) =====
    # `tan * d/dx` (= tan_conv * Dopx) on each of the four field-component diagonal
    # blocks [Ex; Ey; iZHx; iZHy].  Vanishes at slant=0 -> the generator is then
    # byte-identical to the validated VERTICAL operator.  This is what replaces the
    # static ezz*tan^2 / -ezz*tan fold (tan was forced to 0 above), and is what
    # lifts the per-order accuracy from the fold's ~1e-2 floor to the wall-normal
    # ~1e-4 floor at steep slant / strong coupling.
    if abs(tan_conv) > 1e-14:
        for _b in range(4):
            _sl = slice(_b * n, (_b + 1) * n)
            L[_sl, _sl] += tan_conv * Dopx
    return L, n



def _split_modes_flux_metric(W, V, q, lam, n):
    """Forward/backward split by the all-harmonic z-Poynting flux of the genuine
    state.  ``V = -(iZ H)`` (lib-aligned partner), so
    ``Sz = -Im( Ex conj(V[n:]) - Ey conj(V[:n]) )``; the propagating branch keeps
    ``Sz > 0`` (= +z power), evanescent keeps ``Im(q) > 0``."""
    Ex, Ey = W[:n], W[n:]
    Vx, Vy = V[:n], V[n:]                     # -(iZ Hx), -(iZ Hy)
    flux = np.imag(np.sum(Ex * np.conj(Vy) - Ey * np.conj(Vx), axis=0))
    N = 2 * n
    qmax = max(float(np.max(np.abs(q))), 1.0)
    tol = 1e-7 * qmax
    prop = np.abs(q.imag) < tol               # propagating: real q
    fscale = 1e-9 * max(float(np.max(np.abs(flux))), 1.0)
    fwd = np.where(prop, flux > fscale, q.imag > 0.0)
    fidx = np.where(fwd)[0]
    if fidx.size != N:
        score = np.where(prop, flux, q.imag)
        fidx = np.argsort(-score)[:N]
    bidx = np.array(sorted(set(range(len(q))) - set(fidx.tolist())), dtype=int)
    return (W[:, fidx], V[:, fidx], lam[fidx], q[fidx],
            W[:, bidx], V[:, bidx], lam[bidx], q[bidx])



def _layer_modes_metric(mats, k0, slant_angle, kx0=0.0):
    r"""Eigenmodes of the metric generator ``L``: ``L psi = mu psi`` with
    ``mu = -i k0 gamma``, so ``q = gamma = i mu / k0`` and ``lam = -i q =
    mu / k0`` (propagator ``exp(-lam k0 z) = exp(+i q k0 z)``).

    ``kx0`` is the lab transverse incident wavenumber (``k0 Re(n_sup) sin angle``)
    for OBLIQUE incidence; it enters the generator's transverse derivatives and
    the div-conforming Ez bracket (Granet Eq.17).  ``kx0 = 0`` = normal incidence.

    Returns ``(Wf, Vf, lamf, qf, Wb, Vb, lamb, qb)``: ``W = [Ex; Ey]`` (2n) is
    the eigenvector's upper block, and the partner ``V = -G`` where
    ``G = psi[2n:]`` is the genuine magnetic state ``iZ H``.  This state is
    ALREADY lab-Cartesian (the slant is folded entirely into the contravariant
    ``eps^lm/mu^lm``, NOT into the field components), so ``V = -G`` IS the
    genuinely-continuous lab tangential ``[iZ Hx; iZ Hy]`` across the planar
    ``y=const`` interface -- it needs NO inclined->lab shear (a Granet Eq.6 shear
    here would double-count and break continuity; measured directly).  This is the
    uniform gauge sign that aligns the LAYER to the lib half-spaces'
    ``_sem_modes_tensor`` partner convention so they share ONE ``[W; V]``
    continuity in ``_interface_smatrix_general``."""
    L, n = _build_generator_metric(mats, k0, slant_angle, kx0)
    mu, psi = np.linalg.eig(L)               # L psi = mu psi
    q = 1j * mu / k0                          # mu = -i k0 q  -> q = i mu / k0
    lam = -1j * q                             # = mu / k0
    W = psi[:2 * n, :]                        # [Ex; Ey]
    Gpart = psi[2 * n:, :]                    # [iZ Hx; iZ Hy] (PHYSICAL state H)
    V = -Gpart                                # lib-aligned partner sign
    return _split_modes_flux_metric(W, V, q, lam, n)



def _half_M_sym_metric(W, V):
    """Symmetric (vertical homogeneous) half-space field-mode matrix
    ``[[W, W], [V, -V]]`` -- the +/-q convention :func:`_sem_modes_tensor`
    returns for a uniform medium."""
    return np.block([[W, W], [V, -V]])



def _pmm_jones_slant_core(mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max,
                          period, depth, wl, slant_angle, kx0, far_field_orders,
                          label):
    """Shared slanted-tensor far-field core for the BINARY and SEGMENTS solvers.
    Takes the prebuilt nodal grids (the only thing that differs between binary
    and multi-region) and runs: metric-generator layer modes + the lib's PROVEN
    ``_sem_modes_tensor`` lab half-spaces + generalized S-matrix + lab Rayleigh
    far field.  Returns ``(orders, R(2,N), T(2,N), jones(2,2), n_glob)`` in the
    PUBLIC ``exp(-i w t)`` convention (row/col 0 = incident ``E_x``, 1 = ``E_y``).
    ``kx0`` is the oblique Bloch wavenumber; the layer ``V = -G`` partner is
    already lab-Cartesian so it matches the lab half-spaces with no shear."""
    k0 = 2.0 * np.pi / wl
    n_glob = mats["n_glob"]
    Wf_l, Vf_l, lamf_l, _qf, Wb_l, Vb_l, lamb_l, _qb = _layer_modes_metric(
        mats, k0, slant_angle, kx0)
    _geo = _uniform_geo_eig(mats_sup, k0, kx0)
    Ws, Vs, _ls, _qs = _sem_modes_uniform(mats_sup, k0, kx0, eps_sup, _geo)
    Wsub, Vsub, _lb, _qb2 = _sem_modes_uniform(mats_sub, k0, kx0, eps_sub,
                                               _geo)
    Ms = _half_M_sym_metric(Ws, Vs)
    Mb = _half_M_sym_metric(Wsub, Vsub)
    Ml = np.block([[Wf_l, Wb_l], [Vf_l, Vb_l]])

    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: resolution too low to resolve the {2 * m_prop + 1} "
            f"propagating orders (n_glob={n_glob}); raise degree or "
            f"elements_per_region.")
    half = (n_proj - 1) // 2
    orders = np.arange(-half, half + 1)
    G = 2.0 * np.pi / period
    kx = (kx0 + orders * G) / k0              # Bloch-shifted order wavenumbers
    N = len(orders)
    Tp = _sem_fourier_projection(orders, period, mats)

    S = _interface_smatrix_general(Ms, Ml)
    S = _propagation_star_general(S, lamf_l, lamb_l, k0 * depth)
    S = _redheffer_star(S, _interface_smatrix_general(Ml, Mb))
    S11, _S12, S21, _S22 = S

    def _proj_fwd(M):
        Wf = M[:2 * n_glob, :2 * n_glob]
        return np.vstack([Tp @ Wf[:n_glob, :], Tp @ Wf[n_glob:, :]])
    Hsup = _proj_fwd(Ms)
    Hsub = _proj_fwd(Mb)

    # Gain / non-propagating incidence-medium guard (rcwa audit-P1 mirror;
    # PUBLIC-convention eps_sup here -> conj to the guard's internal gauge).
    _require_propagating_incidence(label, np.conj(_C(eps_sup)),
                                   (kx0 / k0) ** 2)
    kz_sup = _kz_forward(eps_sup, kx)
    kz_sub = _kz_forward(eps_sub, kx)
    kx0n = kx0 / k0
    kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0n]))[0]))
    R_eff, T_eff, jones = _assemble_jones_farfield(
        Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc, kx0n, N)
    return orders, R_eff, T_eff, jones, n_glob



def _pmm_jones_slant_solve(period, eps_ridge3, eps_groove3, n_sub, n_sup, depth,
                           duty, wl, slant_angle, degree, n_ridge_el,
                           n_groove_el, grade, far_field_orders, angle=0.0):
    """Single-degree slanted-tensor coupled PMM solve from the genuine metric
    generator (the slanted layer) + homogeneous half-spaces (the lib's PROVEN
    :func:`_sem_modes_tensor` modes) + generalized S-matrix + lab-frame Rayleigh
    far field.

    ``angle`` is the incidence angle (radians); it sets the lab transverse Bloch
    wavenumber ``kx0 = k0 Re(n_sup) sin(angle)`` that enters the div-conforming
    generator, the half-space modes, and the lab Rayleigh projection.  ``angle =
    0`` (normal incidence) is byte-identical to the prior signature.  Combined
    oblique + slant is validated (the div-conforming Ez closure removes the
    Bloch-amplified Liu-2015 spurious null); the layer + lab half-spaces conserve
    energy ~1e-13 and the far field is degree-clean at all slants.

    Returns ``(orders, R(2,N), T(2,N), jones(2,2), n_glob)`` in the PUBLIC
    ``exp(-i w t)`` convention (row/column 0 = incident ``E_x``, 1 = incident
    ``E_y``; ``jones`` columns are the zeroth-order reflected ``[E_x; E_y]``)."""
    er = np.asarray(eps_ridge3, dtype=_C)
    eg = np.asarray(eps_groove3, dtype=_C)
    eps_sup, eps_sub = _C(n_sup) ** 2, _C(n_sub) ** 2
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(_C(n_sup))) * np.sin(float(angle)) * k0
    d_wall = duty * period

    mats = _build_nodal_metric(period, d_wall, _t3_slant(er), _t3_slant(eg),
                               degree, n_ridge_el, n_groove_el, grade)
    # HALF-SPACES: homogeneous (no walls -> no slant); use the lib's PROVEN
    # _sem_modes_tensor modes + far-field plumbing.  The genuine generator is
    # used ONLY for the slanted LAYER; its V is aligned to the lib partner sign
    # (V := -G in _layer_modes_metric) so LAYER + HALF-SPACES share ONE [W; V]
    # continuity convention.
    t_sup = dict(exx=eps_sup, exy=0.0, eyx=0.0, eyy=eps_sup, ezz=eps_sup)
    t_sub = dict(exx=eps_sub, exy=0.0, eyx=0.0, eyy=eps_sub, ezz=eps_sub)
    mats_sup = _build_sem_tensor(period, d_wall, t_sup, t_sup, degree,
                                 n_ridge_el, n_groove_el, grade)
    mats_sub = _build_sem_tensor(period, d_wall, t_sub, t_sub, degree,
                                 n_ridge_el, n_groove_el, grade)
    n_max = max(np.real(n_sup), np.real(n_sub), np.real(np.sqrt(er[0, 0])),
                np.real(np.sqrt(er[1, 1])), np.real(np.sqrt(eg[0, 0])),
                np.real(np.sqrt(eg[1, 1])))
    return _pmm_jones_slant_core(
        mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max, period, depth, wl,
        slant_angle, kx0, far_field_orders, "pmm_jones_1d_slanted")



# ===========================================================================
# COVARIANT OBLIQUE-COORDINATE slant path (Li 1999 JOSA A 16:2521).
# ---------------------------------------------------------------------------
# The convection slant path (_build_generator_metric) carries the tilt as
# `tan d/dx` on the LAB-Cartesian generator, which DIFFERENTIATES the
# discontinuous wall-normal E -> the TM/p-pol per-order accuracy converges only
# ALGEBRAICALLY (~1e-4 at practical degree, the "slant TM floor").  The
# COVARIANT formulation absorbs the tilt into a constant oblique metric so the
# slanted wall becomes a COORDINATE SURFACE: the discontinuous wall-normal field
# is then handled ALGEBRAICALLY by the Li inverse rule and the across-wall
# derivative only ever hits CONTINUOUS combinations -> SPECTRAL (vertical-grade)
# convergence at slant (the channel converges spectrally where convection only
# converges algebraically -- a real capability, the "structurally impossible"
# verdict is refuted).
# ACCURACY (calibrated 2026-06-09 audit P2-B): self-convergence (covariant vs its
# own high-degree limit, and vs the convection path) reaches ~1e-7 by degree ~24,
# BUT that figure is SELF-REFERENTIAL -- the convection path shares the identical
# wall-normal inverse-rule floor, so agreeing with it does not bound the true
# error.  Vs an INDEPENDENT RCWA full-3x3 z-staircase oracle the wall-normal TM
# channel floors at ~2.5e-3 at slant=45 (a plateau, deg16->28: 2.57e-3->2.47e-3);
# the TE channel is clean (<8e-4).  So the honest headline is ~2.5e-3 (TM) vs
# independent ground truth, not 1e-7.  Energy conserves to ~1e-6 across
# cells x slant 15-60 x normal/oblique.
#
# Gauge: for a slanted LAMELLAR (a-dot=0) the covariant tangential components are
# the LAB tangential (E1=Ex, E3=Ey, H1=Hx, H3=Hy) and the flat z=const interfaces
# ARE coordinate surfaces -> CONFORMAL: match lab [Ex,Ey,Hx,Hy] to homogeneous
# half-spaces built by the SAME covariant generator (NOT the lib _sem_modes_tensor
# half-spaces -- those are a different gauge and give a lossless-trap per-order
# error).  beta (the x2=z*secφ wavenumber) maps to the lab kz_eff = beta*secφ/k0,
# which folds in BOTH layer propagation AND the lateral-shift phase.  The solver
# is internally exp(+iωt) (conjugate of the PUBLIC exp(-iωt)); the public wrapper
# conjugates eps in and the Jones out (efficiencies are |.|^2 -> convention-blind).
# ===========================================================================

def _cov_blocks(mats, slant_angle):
    r"""Li covariant E-matrix Z-blocks (Eq.12 + Eq.20), full ``(3,3)`` tensor
    (IN-PLANE OR OUT-OF-PLANE), sec-stretched oblique metric (Li x1=x periodic,
    x2=z propagation, x3=y invariant).  The OUT-OF-PLANE coupling enters through
    the POINTWISE ezz-Schur composites (Li Eq.10/12, formed in x BEFORE
    bracketing) -- ``a_eff = exx - exz ezx/ezz``, ``b_eff = exy - exz ezy/ezz``,
    ``c_eff = eyx - eyz ezx/ezz``, ``d_eff = eyy - eyz ezy/ezz`` -- which replace
    ``exx/exy/eyx/eyy`` everywhere; the slant fold then gives
    ``eps^11 = a_eff + ezz tan^2``, ``eps^12 = -ezz tanφ secφ``,
    ``eps^22 = ezz sec^2``, and the cross composites ride INSIDE the wall-normal
    inverse bracket as solitary entities (``eps^lm/eps^11``, the LOAD-BEARING
    ``/eps^11`` -- NOT ``/a_eff``; the ``ezz tan^2`` difference is the slant
    cross-pol term).  At off-plane=0 a_eff/b_eff/c_eff/d_eff reduce to
    exx/exy/eyx/eyy and this is BYTE-IDENTICAL to the in-plane form.  Returns
    ``(Z11, Z12, Z21, E22inv, Z13, Z31, Z33)`` (each n x n); the single-x-
    derivative off-plane cross blocks are added in :func:`_cov_generator_4n`."""
    iS0 = _safe_inv(mats["S0"])
    tan = np.tan(slant_angle)
    sec = 1.0 / np.cos(slant_angle)

    def E(fn):
        return iS0 @ _coeff_mass_metric(mats, fn)

    def aeff(t):                                          # ezz-Schur composites
        return t["exx"] - t["exz"] * t["ezx"] / t["ezz"]  # (Li Eq.12, pointwise)

    def beff(t):
        return t["exy"] - t["exz"] * t["ezy"] / t["ezz"]

    def ceff(t):
        return t["eyx"] - t["eyz"] * t["ezx"] / t["ezz"]

    def deff(t):
        return t["eyy"] - t["eyz"] * t["ezy"] / t["ezz"]

    def e11(t):
        return aeff(t) + t["ezz"] * tan * tan             # eps^11 (wall-normal)

    def e12(t):
        return -t["ezz"] * tan * sec                      # eps^12 (slant)

    def e22(t):
        return t["ezz"] * sec * sec                       # eps^22 (longitudinal)
    E11 = _safe_inv(E(lambda t: 1.0 / e11(t)))            # [[1/eps^11]]^-1
    T12 = E(lambda t: e12(t) / e11(t))
    T13 = E(lambda t: beff(t) / e11(t))                   # eps^13/eps^11 (Li Eq.12)
    T31 = E(lambda t: ceff(t) / e11(t))                   # eps^31/eps^11
    Tsch22 = E(lambda t: e22(t) - e12(t) * e12(t) / e11(t))
    EE12 = E11 @ T12
    EE21 = T12 @ E11
    EE13 = E11 @ T13
    EE31 = T31 @ E11
    EE22 = T12 @ E11 @ T12 + Tsch22
    EE22i = _safe_inv(EE22)
    EE33 = (T31 @ E11 @ T13
            + E(lambda t: deff(t) - ceff(t) * beff(t) / e11(t)))
    Z11 = E11 - EE12 @ EE22i @ EE21
    Z12 = EE12 @ EE22i
    Z21 = EE22i @ EE21
    return Z11, Z12, Z21, EE22i, EE13, EE31, EE33



def _cov_generator_4n(mats, k0, slant_angle, kx0=0.0, divconf=False):
    r"""Covariant 4n first-order generator ``M`` (state ``X = [Ey; Hy; Hx; Ex]``
    = Li ``[E3; H3; H1; E1]``) for ``dz X = i M X``.  TM block over ``(H3, E1)``,
    TE block over ``(E3, H1)`` (its metric is smooth -> machine-clean), and the
    in-plane ``exy/eyx`` coupling enters as ``M[H3,E3] = -k0 cosφ E13`` and
    ``M[H1,E1] = +k0 cosφ E31`` with ``E33`` replacing ``eyy`` in ``M[H1,E3]``.
    ``kx0`` is the PHYSICAL oblique Bloch wavenumber (``k0 Re(n_sup) sinθ``).
    ``divconf`` selects the longitudinal Ez closure ``M[E1, H3]``: ``False`` ->
    the MODAL ``G (eps^22)^-1 G`` form (byte-identical to the prior in-plane
    covariant); ``True`` -> the DIV-CONFORMING ``INT(1/ezz) B'B'`` form (Granet
    2023 Eq.16-18) which cures the Liu-2015 harmonic-mean spurious null and takes
    the OUT-OF-PLANE channel to machine precision.  The closure must MATCH across
    each z=const interface, so the caller applies the SAME ``divconf`` to the
    layer AND both homogeneous half-spaces."""
    n = mats["n_glob"]
    I = np.eye(n, dtype=_C)
    cos = np.cos(slant_angle)
    sin = np.sin(slant_angle)
    Dop = _safe_solve(mats["S0"], mats["C"])
    G = kx0 * I + (-1j) * Dop                             # alpha (physical)
    kb = k0 * cos
    Z11, Z12, Z21, Z22i, Z13, Z31, Z33 = _cov_blocks(mats, slant_angle)
    M = np.zeros((4 * n, 4 * n), dtype=_C)

    def put(bi, bj, blk):
        M[bi * n:(bi + 1) * n, bj * n:(bj + 1) * n] = blk
    put(0, 0, sin * G)
    put(0, 2, kb * I)
    put(2, 0, kb * Z33 - (cos / k0) * (G @ G))
    put(2, 2, sin * G)
    put(1, 1, -Z12 @ G)
    put(1, 3, -kb * Z11)
    if divconf:
        # DIV-CONFORMING longitudinal Ez closure: INT(1/ezz) B'B' (Granet 2023
        # Eq.16-18) -- machine-precision out-of-plane.  The kx0 terms inject the
        # oblique Bloch shift into the longitudinal derivative.  The cos/k0 factor
        # (vs the convection path's 1/k0) is the oblique-frame z-metric (kz =
        # beta / cos / k0 read-out).
        iS0 = _safe_inv(mats["S0"])
        Lez = mats["L_inv_ezz"].copy()
        if kx0:
            Cas = mats["C_inv_ezz"] - mats["C_inv_ezz"].T
            Lez = Lez - 1j * kx0 * Cas + (kx0 * kx0) * mats["mass"]["inv_ezz"]
        put(3, 1, (cos / k0) * (iS0 @ Lez) - kb * I)
    else:
        put(3, 1, G @ (Z22i / (k0 * cos)) @ G - kb * I)   # MODAL (in-plane)
    put(3, 3, -G @ Z21)
    put(1, 0, -kb * Z13)                                  # TM<-TE coupling
    put(2, 3, kb * Z31)                                   # TE<-TM coupling
    # ----- FULL-3x3 OUT-OF-PLANE single-x-derivative cross blocks -----------
    # Eliminating Ez through (eps^33)^-1 = inv([[ezz]]) adds these (Li Eq.19's
    # alpha acting on the eps^13/eps^23 columns).  Each rides the BARE oblique-
    # frame wall-normal derivative ``cos*Dop`` -- the ``cos`` is the conical
    # projection ``k0bar/k0`` (Li Eq.18) that every z-derivative block carries,
    # and the bare ``Dop`` (NOT the Floquet operator ``G``) is the correct
    # single-derivative operator.  The Hx<-Hy (eyz) block carries a relative
    # minus (the iZHx vs iZHy equation sign).  ALL vanish at off-plane=0
    # (exz=eyz=ezx=ezy=0), so the in-plane generator is byte-identical.
    def _maxoff(key):
        return max(abs(complex(t[key])) for (_xl, _xr, t) in mats["elem_bnds"])
    if any(_maxoff(kk) > 0.0 for kk in ("exz", "eyz", "ezx", "ezy")):
        iS0 = _safe_inv(mats["S0"])

        def Op(fn):
            return iS0 @ _coeff_mass_metric(mats, fn)
        # POINTWISE ezz-ratio composites (Li Eq.12 discipline, same as
        # _cov_blocks: form the ratio in x BEFORE bracketing).  The spectral
        # product [[exz]] @ inv([[ezz]]) of two DISCONTINUOUS factors is the
        # wrong factorization order at material walls (the bare exz/ezx
        # sub-channel's historical ~5e-2..1e-1 TM floor); the single
        # pointwise mass [[exz/ezz]] is the correct composite.  For uniform
        # media the two are identical, so the exact-dispersion anchor below
        # is unchanged.
        XZZ = Op(lambda t: t["exz"] / t["ezz"])
        ZZX = Op(lambda t: t["ezx"] / t["ezz"])
        YZZ = Op(lambda t: t["eyz"] / t["ezz"])
        ZZY = Op(lambda t: t["ezy"] / t["ezz"])
        # conical-projected single x-derivative.  OBLIQUE incidence: the transverse
        # derivative on the periodic envelope is d/dx + i*kx0 (field ~ e^{i kx0 x}
        # u(x), Granet Eq.17), the SAME Bloch shift the convection generator injects
        # via Dopx (_build_generator_metric).  At kx0=0 this is bare Dop, so normal
        # incidence stays byte-identical.
        cD = cos * (Dop + 1j * kx0 * I)
        # FACTOR-i FIX (AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14, lockstep
        # with the metric generator above and rcwa's
        # _layer_eigenmodes_tensor): the off-plane cross blocks carry
        # relative +/-i factors (H-rows x +i, E-rows x -i on the legacy
        # terms).  Pinned by EXACT DISPERSION on uniform OOP slabs: for a
        # uniform medium this generator is a polynomial in Dop, so eig(M)
        # must equal the union over the alpha spectrum of the exact
        # det(kk^T - |k|^2 I + eps) = 0 roots mapped through
        # beta = kz*k0*cos(phi) + alpha*sin(phi).  This assignment is the
        # UNIQUE one of all 256 per-block {+-1, +-i} combos that closes
        # (4e-12 full-spectrum at slant 0/30/45, generic AND symmetric
        # tensors, BOTH Ez closures; next-best combo 1.6e-2).
        M[n:2 * n, n:2 * n] += 1j * (XZZ @ cD)                    # Hy<-Hy (exz)
        M[3 * n:4 * n, 3 * n:4 * n] += -1j * (-cD @ ZZX)          # Ex<-Ex (ezx)
        M[2 * n:3 * n, n:2 * n] += 1j * (-YZZ @ cD)               # Hx<-Hy (eyz)
        M[3 * n:4 * n, 0:n] += -1j * (-cD @ ZZY)                  # Ex<-Ey (ezy)
    return M, n



def _cov_layer_4n(mats, k0, slant_angle, kx0=0.0, divconf=False):
    r"""Covariant layer eigenmodes.  Returns ``(W, V, kz, fwd)`` where
    ``W = [Ex; Ey]`` (2n), ``V = [Hx; Hy]`` (2n), ``kz`` the LAB z-wavenumber /
    ``k0`` (propagator ``exp(i kz k0 z)``; ``kz = beta secφ / k0``), and ``fwd``
    the +z-Poynting / decaying forward mask.  ``divconf`` selects the longitudinal
    closure (see :func:`_cov_generator_4n`); it must be the SAME for the layer and
    both half-spaces of a solve."""
    M, n = _cov_generator_4n(mats, k0, slant_angle, kx0, divconf)
    beta, X = np.linalg.eig(M)
    kz = beta / np.cos(slant_angle) / k0
    Ey, Hy, Hx, Ex = X[:n], X[n:2 * n], X[2 * n:3 * n], X[3 * n:]
    W = np.vstack([Ex, Ey])
    V = np.vstack([Hx, Hy])
    Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
    prop = np.abs(np.imag(kz)) < 1e-7 * max(float(np.max(np.abs(kz))), 1.0)
    fwd = np.where(prop, Sz > 0.0, np.imag(kz) > 0.0)
    return W, V, kz, fwd



def _cov_split(W, V, kz, fwd):
    """Forward/backward split, exactly half each (by Poynting / decay).

    When the raw forward mask already has exactly ``half`` members, use it
    verbatim (the default slant path).  Otherwise fall back to a forward-ness
    score that must be both SIGN- and SCALE-consistent (audit P2-C / followup F5):
    propagating modes are ranked by their z-Poynting flux ``Sz`` (~length^2) and
    evanescent modes by ``Im(kz)`` (dimensionless) -- two INCOMMENSURATE scales.
    Each population is NORMALIZED to unit max magnitude before the merge (mirroring
    the normalized RCWA analog), so an out-of-scale ``Sz`` cannot dominate the
    rank purely by units in this rebalance branch."""
    n = W.shape[0] // 2
    half = W.shape[1] // 2
    if int(np.sum(fwd)) == half:
        fidx = np.where(fwd)[0]
    else:
        Ex, Ey, Hx, Hy = W[:n], W[n:], V[:n], V[n:]
        Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
        imk = np.imag(kz)
        prop = np.abs(imk) < 1e-7 * max(float(np.max(np.abs(kz))), 1.0)
        # normalize each population to unit max magnitude so the propagating (Sz)
        # and evanescent (Im kz) criteria are comparable across their unit gap
        sz_sc = (max(float(np.max(np.abs(Sz[prop]))), 1e-300)
                 if np.any(prop) else 1.0)
        imk_sc = (max(float(np.max(np.abs(imk[~prop]))), 1e-300)
                  if np.any(~prop) else 1.0)
        sc = np.where(prop, Sz / sz_sc, imk / imk_sc)
        fidx = np.argsort(-sc)[:half]
    bidx = np.array(sorted(set(range(W.shape[1])) - set(fidx.tolist())),
                    dtype=int)
    return fidx, bidx



def _pmm_jones_oblique_core(mats, mats_s, mats_b, eps_sup, eps_sub, n_max,
                           period, depth, wl, slant_angle, kx0, far_field_orders,
                           label):
    """Covariant slanted Jones far field (covariant half-spaces).  All eps are
    INTERNAL (conjugate) convention; the caller conjugates the returned Jones.
    Returns ``(orders, R(2,N), T(2,N), jones(2,2), n_glob)``."""
    k0 = 2.0 * np.pi / wl
    n = mats["n_glob"]
    # DIV-CONFORMING Ez closure when the LAYER is out-of-plane (machine-precision
    # OOP); applied to the layer AND both half-spaces so the closure form MATCHES
    # across every z=const interface (the S-matrix continuity demands it).  An
    # in-plane layer keeps the MODAL closure (byte-identical to the prior path).
    divconf = any(max(abs(complex(t[kk])) for (_xl, _xr, t) in mats["elem_bnds"])
                  > 0.0 for kk in ("exz", "eyz", "ezx", "ezy"))
    Wl, Vl, kzl, fwl = _cov_layer_4n(mats, k0, slant_angle, kx0, divconf)
    Ws, Vs, kzs, fws = _cov_layer_4n(mats_s, k0, slant_angle, kx0, divconf)
    Wb, Vb, kzb, fwb = _cov_layer_4n(mats_b, k0, slant_angle, kx0, divconf)
    fl, bl = _cov_split(Wl, Vl, kzl, fwl)
    fs, _bs = _cov_split(Ws, Vs, kzs, fws)
    fb, _bb = _cov_split(Wb, Vb, kzb, fwb)

    def Mmat(W, V, f, b):
        return np.block([[W[:, f], W[:, b]], [V[:, f], V[:, b]]])

    def Msym(W, V, f):
        return np.block([[W[:, f], W[:, f]], [V[:, f], -V[:, f]]])
    Ms = Msym(Ws, Vs, fs)
    Ml = Mmat(Wl, Vl, fl, bl)
    Mb = Msym(Wb, Vb, fb)
    S = _interface_smatrix_general(Ms, Ml)
    S = _propagation_star_general(S, -1j * kzl[fl], -1j * kzl[bl], k0 * depth)
    S = _redheffer_star(S, _interface_smatrix_general(Ml, Mb))
    S11, _S12, S21, _S22 = S

    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n if n % 2 else n - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: resolution too low to resolve the {2 * m_prop + 1} "
            f"propagating orders (n_glob={n}); raise degree or "
            f"elements_per_region.")
    half = (n_proj - 1) // 2
    orders = np.arange(-half, half + 1)
    N = len(orders)
    kx = kx0 / k0 + orders * (2.0 * np.pi / period) / k0   # kx / k0 per order
    Tp = _sem_fourier_projection(orders, period, mats)
    m0 = int(np.where(orders == 0)[0][0])

    def kz_ord(eps):
        # PUBLIC-convention forward kz (Re>=0) for the far-field FLUX weight + the
        # propagating mask in _assemble_jones_farfield.  ``eps`` arrives in the
        # INTERNAL exp(+iwt) (conjugated) gauge; un-conjugating it restores
        # Re(kz)>=0 for a forward wave into a LOSSY exit half-space -- the raw
        # internal kz has Re<0 there, which the ``Re(kz)>0`` mask silently zeroes
        # (T=0).  Lossless eps is real -> conj is a no-op -> byte-unchanged.  P1-A.
        return _kz_forward(np.conj(_C(eps)), kx)
    # Gain / non-propagating incidence-medium guard (rcwa audit-P1 mirror).
    # ``eps_sup`` is already INTERNAL exp(+iwt) here -- exactly the gauge the
    # rcwa guard expects -- so it is passed WITHOUT the public->internal conj.
    _require_propagating_incidence(label, _C(eps_sup), (kx0 / k0) ** 2)
    kzo_s = kz_ord(eps_sup)
    kzo_b = kz_ord(eps_sub)
    kz_inc = float(np.real(kzo_s[m0]))
    Hsup = np.vstack([Tp @ Ws[:n, fs], Tp @ Ws[n:, fs]])
    Hsub = np.vstack([Tp @ Wb[:n, fb], Tp @ Wb[n:, fb]])
    R, T, jones = _assemble_jones_farfield(
        Hsup, Hsub, S11, S21, orders, kx, kzo_s, kzo_b, kz_inc, kx0 / k0, N)
    return orders, R, T, jones, n



def _pmm_jones_oblique_solve(period, eps_ridge3, eps_groove3, n_sub, n_sup,
                             depth, duty, wl, slant_angle, degree, n_ridge_el,
                             n_groove_el, grade, far_field_orders, angle=0.0):
    """Covariant oblique-coordinate slanted-Jones solve (the SPECTRAL slant path;
    ``factorization='covariant'``).  Same signature/returns as
    :func:`_pmm_jones_slant_solve`; converges spectrally where the convection
    path converges algebraically (same physical answer).  Full ``(3, 3)`` tensors
    IN-PLANE OR OUT-OF-PLANE (the out-of-plane coupling enters the covariant
    generator via the Li Eq.12 ezz-Schur composites + cos*Dop cross blocks)."""
    er = np.conj(np.asarray(eps_ridge3, dtype=_C))        # exp(-iωt) -> internal
    eg = np.conj(np.asarray(eps_groove3, dtype=_C))
    eps_sup = np.conj(_C(n_sup) ** 2)
    eps_sub = np.conj(_C(n_sub) ** 2)
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(np.conj(_C(n_sup)))) * np.sin(float(angle)) * k0
    d_wall = duty * period
    mats = _build_nodal_metric(period, d_wall, _t3_slant(er), _t3_slant(eg),
                               degree, n_ridge_el, n_groove_el, grade)
    ts = dict(exx=eps_sup, exy=0.0, eyx=0.0, eyy=eps_sup, ezz=eps_sup,
              exz=0.0, eyz=0.0, ezx=0.0, ezy=0.0)
    tb = dict(exx=eps_sub, exy=0.0, eyx=0.0, eyy=eps_sub, ezz=eps_sub,
              exz=0.0, eyz=0.0, ezx=0.0, ezy=0.0)
    mats_s = _build_nodal_metric(period, d_wall, ts, ts, degree, n_ridge_el,
                                 n_groove_el, grade)
    mats_b = _build_nodal_metric(period, d_wall, tb, tb, degree, n_ridge_el,
                                 n_groove_el, grade)
    n_max = max(np.real(n_sup), np.real(n_sub),
                np.real(np.sqrt(er[0, 0])), np.real(np.sqrt(er[1, 1])),
                np.real(np.sqrt(eg[0, 0])), np.real(np.sqrt(eg[1, 1])))
    orders, R, T, jones, ng = _pmm_jones_oblique_core(
        mats, mats_s, mats_b, eps_sup, eps_sub, n_max, period, depth, wl,
        slant_angle, kx0, far_field_orders, "pmm_jones_1d_slanted")
    return orders, R, T, np.conj(jones), ng        # internal -> public exp(-iωt)



def _pmm_jones_oblique_segments_solve(period, widths, seg_tensors3, n_sub, n_sup,
                                      depth, wl, slant_angle, degree,
                                      n_el_per_region, grade, far_field_orders,
                                      angle=0.0):
    """N-region covariant oblique-coordinate slanted-Jones solve -- the multi-
    region generalization of :func:`_pmm_jones_oblique_solve`.  The covariant
    generator + far field are region-count-agnostic; only the nodal grid differs
    (the segment element table, and the homogeneous half-spaces on the SAME
    segment grid).  Full ``(3, 3)`` tensors IN-PLANE OR OUT-OF-PLANE (the
    out-of-plane coupling enters the covariant generator via the Li Eq.12
    ezz-Schur composites + cos*Dop cross blocks, same as the binary path)."""
    # `_segment_elem_bnds` lays the regions out REVERSED in x ([::-1]); the
    # convection core un-mirrors via the lib half-space/far-field handedness, but
    # the covariant far-field does not -- so PRE-REVERSE widths + tensors here so
    # the internal [::-1] cancels and the covariant solves the FORWARD layout
    # (region 0 at x=0), reporting orders in the user's input frame.  (At slant=0
    # the mirror is an order swap m<->-m; at slant!=0 it also flips φ -> -φ.)
    widths = list(widths)[::-1]
    seg_tensors3 = list(seg_tensors3)[::-1]
    er_seg = [np.conj(np.asarray(t, dtype=_C)) for t in seg_tensors3]  # -> internal
    eps_sup = np.conj(_C(n_sup) ** 2)
    eps_sub = np.conj(_C(n_sub) ** 2)
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(np.conj(_C(n_sup)))) * np.sin(float(angle)) * k0
    nseg = len(widths)
    seg_t = [_t3_slant(t) for t in er_seg]
    mats = _build_nodal_metric_segments(period, widths, seg_t, degree,
                                        n_el_per_region, grade)
    ts = _t3_slant(np.diag([eps_sup, eps_sup, eps_sup]).astype(_C))
    tb = _t3_slant(np.diag([eps_sub, eps_sub, eps_sub]).astype(_C))
    mats_s = _build_nodal_metric_segments(period, widths, [ts] * nseg, degree,
                                          n_el_per_region, grade)
    mats_b = _build_nodal_metric_segments(period, widths, [tb] * nseg, degree,
                                          n_el_per_region, grade)
    n_max = max([np.real(n_sup), np.real(n_sub)]
                + [np.real(np.sqrt(t[0, 0])) for t in er_seg]
                + [np.real(np.sqrt(t[1, 1])) for t in er_seg])
    orders, R, T, jones, ng = _pmm_jones_oblique_core(
        mats, mats_s, mats_b, eps_sup, eps_sub, n_max, period, depth, wl,
        slant_angle, kx0, far_field_orders, "pmm_jones_1d_slanted_segments")
    return orders, R, T, np.conj(jones), ng        # internal -> public exp(-iωt)



def _pmm_jones_slant_segments_solve(period, widths, seg_tensors3, n_sub, n_sup,
                                    depth, wl, slant_angle, degree,
                                    n_el_per_region, grade, far_field_orders,
                                    angle=0.0):
    """N-region slanted-tensor coupled PMM solve -- the multi-region
    generalization of :func:`_pmm_jones_slant_solve`.  The metric generator and
    the lab-frame far field are region-count-agnostic; only the nodal grid
    differs (the segment element table from :func:`_build_nodal_metric_segments`,
    and the homogeneous half-spaces on the SAME segment grid).  ``seg_tensors3[i]``
    = ``(3, 3)`` in-plane tensor of region ``i``, ``widths[i]`` its fractional
    width.  Returns ``(orders, R(2,N), T(2,N), jones(2,2), n_glob)`` in the same
    convention as :func:`_pmm_jones_slant_solve`."""
    eps_sup, eps_sub = _C(n_sup) ** 2, _C(n_sub) ** 2
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(_C(n_sup))) * np.sin(float(angle)) * k0
    seg_t = [_t3_slant(np.asarray(t, dtype=_C)) for t in seg_tensors3]
    nseg = len(widths)
    mats = _build_nodal_metric_segments(period, widths, seg_t, degree,
                                        n_el_per_region, grade)
    t_sup = dict(exx=eps_sup, exy=0.0, eyx=0.0, eyy=eps_sup, ezz=eps_sup)
    t_sub = dict(exx=eps_sub, exy=0.0, eyx=0.0, eyy=eps_sub, ezz=eps_sub)
    mats_sup = _build_sem_tensor_segments(period, widths, [t_sup] * nseg, degree,
                                          n_el_per_region, grade)
    mats_sub = _build_sem_tensor_segments(period, widths, [t_sub] * nseg, degree,
                                          n_el_per_region, grade)
    arrs = [np.asarray(t, dtype=_C) for t in seg_tensors3]
    n_max = max([np.real(n_sup), np.real(n_sub)]
                + [np.real(np.sqrt(t[0, 0])) for t in arrs]
                + [np.real(np.sqrt(t[1, 1])) for t in arrs])
    return _pmm_jones_slant_core(
        mats, mats_sup, mats_sub, eps_sup, eps_sub, n_max, period, depth, wl,
        slant_angle, kx0, far_field_orders, "pmm_jones_1d_slanted_segments")



def _pmm_jones_slant_diag_solve(period, er, eg, n_sub, n_sup, depth, duty, wl,
                                slant_angle, degree, elements_per_region, grade,
                                far_field_orders, angle=0.0):
    r"""DIAGONAL-tensor slanted-Jones solve via the DIV-CONFORMING scalar slant
    operator (THE DIAGONAL CURE, round 16).

    For a diagonal in-plane tensor (``exy = eyx = 0``) WITH ``exx == ezz`` the
    TE / TM channels DECOUPLE and the scalar :func:`_pmm_slant_solve`
    (``_sem_modes_slant``, which puts ``1/eps`` INSIDE the z-stiffness -- a
    div-conforming, Liu-2015-spurious-mode-free TM operator) is EXACT per
    channel:

      * TE sees ``eyy``          -> ``n = sqrt(eyy)``  (row 1, incident ``E_y``)
      * TM sees ``exx`` (= ezz)  -> ``n = sqrt(exx)``  (row 0, incident ``E_x``)

    The off-diagonal Jones is identically zero (no ``E_x`` <-> ``E_y`` coupling).
    Returns ``(orders, R_eff(2, M), T_eff(2, M), jones(2, 2))`` in the EXACT
    :func:`_pmm_jones_slant_solve` convention (row/col 0 = incident ``E_x`` = TM,
    1 = incident ``E_y`` = TE; ``jones`` columns are the zeroth-order reflected
    ``[E_x; E_y]``, PUBLIC ``exp(-i w t)``, no conjugation).

    See Granet 2017 (JOSA A 34:975) / Granet 2023 for the scalar slant operator
    and Liu 2015 (CiCP 18:467) for the divergence (Gauss-law) condition the
    pointwise metric-generator Ez-elimination violates."""
    # DEFENSE-IN-DEPTH (fail loud, never silently wrong): the scalar-channel cure is
    # EXACT only inside its validity domain -- a diagonal tensor (exy=eyx=0) with
    # exx==ezz, at normal incidence OR on a vertical grating (NOT combined oblique +
    # slant, whose scalar per-order split is wrong while energy still conserves --
    # the lossless trap that defeats an energy assertion).  The dispatch in
    # pmm_jones_1d_slanted already enforces exactly this domain before routing here;
    # the cure RE-ASSERTS it so any future mis-route or direct call raises instead
    # of returning a wrong answer (a missing off-diagonal Jones, a wrong TM index,
    # or a ~10%-wrong oblique+slant split).
    er = np.asarray(er, dtype=_C)
    eg = np.asarray(eg, dtype=_C)
    _scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    _inplane_off = max(abs(er[0, 1]), abs(er[1, 0]), abs(eg[0, 1]), abs(eg[1, 0]))
    # |exx - ezz| per region: nonzero means the diagonal-cure precondition
    # (exx == ezz) is violated, so route to the metric generator instead.
    _exx_minus_ezz = max(abs(er[0, 0] - er[2, 2]), abs(eg[0, 0] - eg[2, 2]))
    if _inplane_off > 1e-9 * _scale or _exx_minus_ezz > 1e-9 * _scale:
        raise RuntimeError(
            "pmm_jones_1d_slanted (diagonal cure): the scalar-channel cure requires "
            "a DIAGONAL in-plane tensor (exy=eyx=0) with exx==ezz in both regions; "
            "a coupled or exx!=ezz tensor must route through the metric generator "
            "-- internal dispatch invariant violated.")
    if abs(float(angle)) > 1e-12 and abs(float(slant_angle)) > 1e-12:
        raise RuntimeError(
            "pmm_jones_1d_slanted (diagonal cure): combined oblique incidence + "
            "nonzero slant must route through the metric generator, not the scalar "
            "diagonal cure (its per-order split is wrong for that combination) -- "
            "internal dispatch invariant violated.")
    kw = dict(angle=float(angle), elements_per_region=int(elements_per_region),
              grade=bool(grade), far_field_orders=int(far_field_orders),
              degree=int(degree), return_coeffs=True)
    # TM channel: wall-normal exx (== ezz, asserted by the caller's dispatch).
    o_tm, R_tm, T_tm, _ngt, r0_tm, _t0_tm = _pmm_slant_solve(
        period, np.sqrt(er[0, 0]), np.sqrt(eg[0, 0]), n_sub, n_sup, depth, duty,
        wl, slant_angle, polarization="tm", **kw)
    # TE channel: wall-tangential eyy.
    o_te, R_te, T_te, _nge, r0_te, _t0_te = _pmm_slant_solve(
        period, np.sqrt(er[1, 1]), np.sqrt(eg[1, 1]), n_sub, n_sup, depth, duty,
        wl, slant_angle, polarization="te", **kw)
    # The two channels share period/depth/duty/slant/degree/etc, so the retained
    # Rayleigh-order set is identical; guard the invariant explicitly.
    if not np.array_equal(o_tm, o_te):
        raise RuntimeError(
            "pmm_jones_1d_slanted (diagonal cure): the TE/TM scalar-slant order "
            "sets disagree -- internal invariant violated.")
    orders = o_tm
    M = orders.size
    R_eff = np.zeros((2, M))
    T_eff = np.zeros((2, M))
    R_eff[0], T_eff[0] = R_tm, T_tm          # row 0 = incident E_x = TM
    R_eff[1], T_eff[1] = R_te, T_te          # row 1 = incident E_y = TE
    jones = np.zeros((2, 2), _C)
    jones[0, 0] = r0_tm                       # diag(r0_TM, r0_TE); off-diag = 0
    jones[1, 1] = r0_te
    return orders, R_eff, T_eff, jones


# ===========================================================================
# ARCHIVE -- superseded methods, preserved with the WHY (NOT executed)
# ===========================================================================
# The PMM 1-D solver grew through many incremental rounds.  Retired approaches
# are kept here as raw strings (parsed, never run or linted) so the institutional
# record -- what was tried and WHY it was superseded -- survives in the code, not
# only in volatile notes.

_ARCHIVE_SLANT_FOLD = r'''
SUPERSEDED: the ezz*tan^2 STATIC METRIC FOLD for the slant (round 11; replaced by
the exact tan_conv*Dopx CONVECTION in _build_generator_metric, 2026-06-07).

WHAT IT COMPUTED -- the Edee-Granet 2024 contravariant metric fold for a slant
phi, formerly in _build_generator_metric behind `if abs(tan) < 1e-14: ... else:`
arms (tan was later hard-set to 0, making the else-arms dead; this is them):

    # in-plane wall-normal + cross blocks (the dead else-arm):
    Oeps11 = _build_inv_rule_metric(
        mats, lambda t_: 1.0 / (t_["exx"] + t_["ezz"] * tan * tan), iS0)  # eps^11 = exx + ezz tan^2
    Oeps13 = iS0 @ _coeff_mass_metric(mats, lambda t_: -t_["ezz"] * tan)  # eps^13 = eps^31 = -ezz tan
    Oeps31 = Oeps13.copy()
    # OOP cross-terms (the dead else-arm); raw ezz (slant is a metric fold here):
    Oeps13 = iS0 @ _coeff_mass_metric(mats, lambda t_: -t_["ezz"] * tan)
    Oeps31 = Oeps13.copy()
    # mu fold:
    Mu11 = sec2 * I          # mu^11 = sec^2 (sec2 = 1/cos(phi)^2)
    Mu13 = -tan * I          # mu^13 = mu^31 = -tan
    Mu31 = -tan * I

    # the now-retired helper the fold used:
    def _build_inv_rule_metric(mats, inv_fn, iS0):
        """Li inverse-rule operator [[coeff]]^-1: direct mass of 1/coeff, iS0, invert."""
        M = _coeff_mass_metric(mats, inv_fn)
        return _safe_inv(iS0 @ M)

WHY IT WAS TRIED: the original (round-11) slant realization.  The contravariant
fold sqrt(g) J^-1 eps J^-T with J = [[1,0,tan],[0,1,0],[0,0,1]] gives
eps^11 = exx + ezz tan^2, eps^13 = eps^31 = -ezz tan, mu^11 = sec^2,
mu^13 = mu^31 = -tan.

LOAD-BEARING SIGN DETAIL (do not lose if revisiting): eps^13 and mu^13 must SHARE
the sign of tan -- this is the J^-1 eps J^-T CONTRAVARIANT fold, NOT J eps J^T --
so TE and TM diffract to the SAME side.  The wrong (J eps J^T) fold flips eps^13's
sign so TE and TM convect opposite ways (TE matches the reference at one slant
sign, TM at the other) = unphysical opposite-side diffraction.

WHY SUPERSEDED: the fold caps per-order TM accuracy at ~1e-2 for strongly-coupled
/ steep-slant cells.  The ezz-Schur cancels the slant in the longitudinal then
re-injects it as the STATIC wall-normal ezz*tan^2, whose factorization order is
wrong for the DISCONTINUOUS wall-normal -> per-order TM floors at ~1e-2 (vs the
convection treatment's ~1e-4).  The fix (2026-06-07) carries the slant as the
EXACT first-order convection tan*d/dx (tan_conv*Dopx) added to the CLEAN slant=0
generator, reaching the ~1e-4 wall-normal floor uniformly.  (The genuinely-
covariant Li-1999 oblique-coordinate path, factorization='covariant', converges
SPECTRALLY rather than algebraically by making the wall a coordinate surface --
self-converging to ~1e-7, though vs an INDEPENDENT full-3x3 oracle the TM floor
is ~2.5e-3 at slant=45 / TE <8e-4; see the COVARIANT block above, audit P2-B.)
'''


# Register the PMM module caches with the library cache registry (so the global
# "clear all caches" path empties them too).  Canonical v4.16.0 enrollment pattern
# (mirrors rcwa/_core.py and propagators/propagation.py).
try:
    import sys as _sys

    from ..._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer(
        "pmm_lagrange_dref",
        lambda: getattr(_sys.modules[__name__], "_clear_pmm_caches")(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass

__all__ = [
    "_C",
    "_COV_MIN_SLANT_RAD",
    "_resolve_order_count",
    "_promote_eps_tensor",
    "_resolve_incidence",
    "_resolve_incidence_checked",
    "_forward_branch_flip",
    "_freeze_cached",
    "_mass_flux_cut",
    "_STABILIZE_MAX_SCAN",
    "_MIN_PLATEAU",
    "_PASSIVE_TOL",
    "_CLUSTER_TOL",
    "_PER_ORDER_TOL",
    "_aligned_max_diff",
    "_converged_cluster",
    "_readonly",
    "_gll_nodes_weights",
    "_LAGRANGE_DREF_CACHE",
    "_LAGRANGE_DREF_LOCK",
    "_clear_pmm_caches",
    "_lagrange_derivative_matrix",
    "_graded_boundaries",
    "_build_sem",
    "_ill_scaled",
    "_equil_scale",
    "_safe_inv",
    "_safe_solve",
    "_safe_geig",
    "_sem_modes",
    "_sem_fourier_projection",
    "_interface_smatrix",
    "_propagation_smatrix",
    "_redheffer_star",
    "_kz_forward",
    "_assemble_jones_farfield",
    "_scalar_farfield_RT",
    "_n_propagating_orders",
    "_pmm_solve",
    "_pmm_solve_core",
    "_JONES_PASSIVE_TOL",
    "_build_sem_tensor",
    "_sem_modes_tensor",
    "_sem_modes_uniform",
    "_uniform_geo_eig",
    "_pmm_jones_solve",
    "_pmm_jones_solve_core",
    "_segment_walls",
    "_segment_elem_bnds",
    "_l2g_periodic",
    "_build_sem_segments",
    "_build_sem_tensor_segments",
    "_pmm_solve_segments",
    "_pmm_jones_solve_segments",
    "_lossy_incidence",
    "_stabilize_scalar",
    "_stabilize_jones",
    "_graded_fractions",
    "_jpmm_build_topology",
    "_jpmm_build_static",
    "_jpmm_build_dynamic",
    "_jpmm_order_set",
    "_require_concrete_wavelength",
    "_jpmm_fourier_projection",
    "_jpmm_projection_quad",
    "_jpmm_fourier_projection_jax",
    "_jpmm_assemble",
    "_jpmm_sem_modes",
    "_jpmm_solve",
    "_pmm_efficiency_1d_jax",
    "_jpmm_assemble_tensor",
    "_jpmm_sem_modes_tensor",
    "_jpmm_jones_solve",
    "_pmm_jones_1d_jax",
    "_tensor3_dict",
    "_pmm_union_grid",
    "_build_sem_slant",
    "_sem_modes_slant",
    "_modes_M_slant",
    "_pmm_slant_solve",
    "_t3_slant",
    "_build_nodal_metric",
    "_build_nodal_metric_segments",
    "_coeff_mass_metric",
    "_build_generator_metric",
    "_split_modes_flux_metric",
    "_layer_modes_metric",
    "_half_M_sym_metric",
    "_pmm_jones_slant_core",
    "_pmm_jones_slant_solve",
    "_cov_blocks",
    "_cov_generator_4n",
    "_cov_layer_4n",
    "_cov_split",
    "_pmm_jones_oblique_core",
    "_pmm_jones_oblique_solve",
    "_pmm_jones_oblique_segments_solve",
    "_pmm_jones_slant_segments_solve",
    "_pmm_jones_slant_diag_solve",
]
