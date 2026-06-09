"""
Polynomial Modal Method (PMM) -- 1-D lamellar grating
=====================================================

A NON-Fourier rigorous modal solver for a 1-D binary (lamellar) grating,
complementary to the Fourier Modal Method (RCWA) in
:mod:`~lumenairy.elements.rcwa`.  Instead of expanding the periodic field in a
global Fourier (trigonometric) harmonic basis, PMM expands it in LOCAL
subsectional high-degree polynomials -- one C0 spectral element per homogeneous
subsection (ridge / groove), with the element boundary placed EXACTLY on each
grating wall.  Because each subsection has constant permittivity, every Galerkin
integral is exact and there is NO Gibbs phenomenon at the discontinuities, so
the method converges SPECTRALLY (exponentially) in the polynomial degree.

This is the subsectional-polynomial modal method of Edee (JOSA A 28, 2006
(2011)); the spectral-element (Gauss-Lobatto-Legendre nodal) realisation here
is mathematically equivalent and well-conditioned.

Why PMM over the Fourier method (and over the just-added ASR coordinate
stretch)
--------------------------------------------------------------------------
* **No accuracy floor.**  ASR (``rcwa_efficiency_1d(asr_eta=...)``) plateaus at
  ~1e-4 for TM because its ``u<->x`` Rayleigh coordinate stretch is still a
  global Fourier basis whose dense bridge inherits the Fourier-truncation
  error.  PMM never truncates a Fourier series for the field: ``eps`` is
  piecewise-constant with the discontinuity ON an element boundary, so the
  representation is exact and the TM error drops monotonically with no plateau
  (TE self-converges to ~1e-11).
* **No conditioning ceiling.**  The homogeneous super/substrate regions are
  expanded in the SAME nodal basis (their own uniform-``eps`` spectral
  eigenproblem), so every layer<->region interface is a square, WELL-
  CONDITIONED nodal-space mode match (cond ~ 1).  The Rayleigh plane-wave
  projection is applied ONCE, post-recursion, FORWARD only (a small overlap
  onto the propagating orders) -- it is NEVER inverted as a tall bridge, which
  is exactly what gives ASR/FMM-hybrid schemes their floor.

Conventions (match the rest of the library)
-------------------------------------------
PUBLIC ``exp(-i w t)`` convention end-to-end: ``n = n + i kappa`` (``Im eps >
0`` for loss), forward plane wave ``exp(+i kz z)``, decay ``Im(kz) >= 0``.  No
index conjugation -- the solver is self-contained and validated by EFFICIENCIES
(real power fractions), which are convention-independent.  (Mixing the public
and engineering conventions was the single biggest pitfall during development;
this module deliberately stays in one convention throughout.)

Factorization (the bug-prone part, get it right or TM/metals converge slowly)
-----------------------------------------------------------------------------
The TM operator is built from ``1/eps``, NOT ``eps`` -- the polynomial analogue
of Li's inverse rule.  With ``q = gamma/k0 = n_eff`` the generalized modal
eigenproblems are::

    TE:  ( Peps - L/k0^2  ) v = q^2 S0   v
    TM:  ( S0   - Linv/k0^2) v = q^2 Pinv v

    S0   = INT B_i B_j            Peps = INT eps   B_i B_j
    Pinv = INT (1/eps) B_i B_j    L    = INT B_i' B_j'
    Linv = INT (1/eps) B_i' B_j'

Putting ``1/eps`` on the TM right-mass (``Pinv``) and weighted stiffness
(``Linv``) is the inverse rule (using ``eps`` instead gives the slow Laurent /
algebraic TM rate).  The modal partner (magnetic block for the S-matrix) carries
the per-mode weight ``q`` (TE) or ``q/eps`` i.e. the ``(1/eps)``-weighted field
(TM).  The S-matrix eigenvalue is ``lam = -i q`` with ``Im(q) >= 0`` so the
forward propagator ``exp(-lam k0 L) = exp(+i q k0 L)`` decays.

Capability matrix (current)
---------------------------
The solver now spans the full 1-D family.  Two response kinds x two geometries x
two wall orientations x two anisotropy classes x two backends:

* RESPONSE: scalar efficiencies (:func:`pmm_efficiency_1d`) OR full ``2x2`` Jones
  (:func:`pmm_jones_1d`); :func:`pmm_1d` is the unified facade that auto-routes
  by geometry, and :class:`PMMStack` is the MULTILAYER z-stack entry.
* GEOMETRY: binary (ridge|groove) OR multi-region ``*_segments`` (N graded
  regions on one shared nodal grid).
* WALLS: VERTICAL OR SLANTED (``slant_angle``, ``*_slanted*``); normal OR oblique
  incidence (``angle`` adds the ``+i kx0`` Bloch shift; forward modes chosen by
  the z-Poynting flux).
* ANISOTROPY: scalar/diagonal, full ``(3, 3)`` IN-PLANE tensors (``exy``/``eyx``
  couple ``[E_x; E_y]``), AND OUT-OF-PLANE (``eps_xz/yz/zx/zy``).
* BACKEND: NumPy/SciPy throughout; JAX-differentiable twins for the scalar and
  in-plane vertical Jones paths (slant/OOP JAX is a follow-on).

SLANT has TWO factorizations (kwarg ``factorization``), same physical answer:
  - ``'convection'`` -- the CONVECTION METRIC GENERATOR (lab-Cartesian ``[E;H]``,
    slant carried as ``tan * d/dx`` convection).  Fully general (handles
    OUT-OF-PLANE and mixed-slant stacks) but the TM/p-pol channel converges only
    ALGEBRAICALLY (~1e-4 per-order floor): the convection differentiates the
    discontinuous wall-normal field.  NB energy conserves EXACTLY at every degree
    even while the per-order split is still ~1e-4 off (a lossless cell
    auto-balances total power) -- so do NOT read energy conservation as a
    per-order accuracy proxy.
  - ``'covariant'`` -- the genuinely-covariant Li-1999 OBLIQUE-COORDINATE
    generator (``_cov_*``): the slanted wall becomes a coordinate surface so the
    wall-normal discontinuity is handled algebraically and the TM channel
    converges SPECTRALLY (vertical-grade ~1e-7), ~100-2400x fewer degrees.
    Handles IN-PLANE AND OUT-OF-PLANE (full 3x3) tensors: the out-of-plane
    ``eps_xz/yz/zx/zy`` coupling enters via the pointwise ezz-Schur composites
    (Li-1999 Eq.12) + the ``cos*Dop`` single-x-derivative cross blocks (Li Eq.18/
    19), SPECTRAL at slant -- ~15x fewer degrees than convection for out-of-plane
    too.  (The earlier "covariant cannot do out-of-plane" verdict was wrong: it
    was four operator/factorization bugs -- ``G``-vs-``Dop``, the ``cos`` conical
    projection, the eyz cross-block sign, and the ``/eps^11`` vs ``/a_eff``
    inverse-rule denominator -- all since fixed and Li-grounded.)
  - ``'auto'`` (default) picks covariant for ANY slanted cell (in-plane or
    out-of-plane), convection otherwise (vertical, where the vertical solver is
    already exact).

The Li factorization is realized in the nodal basis: the wall-normal inverse rule
``[[1/exx]]^{-1}`` becomes ``inv(hat(1/exx))`` (the nodal multiply-by-``1/exx``
operator, inverted); the ``Kx``-derivative terms (``Ez``-elimination, ``Kx^2``)
become spectral-element STIFFNESS operators weighted by ``1/ezz`` / ``1`` -- so
the inverse rule is AUTOMATIC and exact (the ``eps`` jump is on an element
boundary).  TM is monotone-no-floor for VERTICAL gratings (the matched partner
``Ex = q (1/eps) Hy`` is discontinuous at the wall and a C0 nodal value averages
that jump; mesh grading toward the walls recovers the corner resolution -- the
speed lever for metal TM).
"""
from __future__ import annotations

import cmath
import functools
import threading
import warnings
from typing import Tuple

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import Legendre

# Backend detection for the JAX (differentiable) dispatch in pmm_efficiency_1d.
# Mirrors rcwa's pattern: a JAX input routes to the self-contained jnp twin,
# while a NumPy input falls through to the original (byte-identical) code.
from ..backend import is_jax_array

# Reused for the slanted-grating solver: the slant breaks the +/-q field
# symmetry (like a full-3x3 tensor layer), so it needs the GENERALIZED
# (explicit forward/backward) S-matrix.  rcwa does NOT import pmm, so this
# top-level import introduces no cycle.
from .rcwa import _interface_smatrix_general, _propagation_smatrix_general

__all__ = ["pmm_efficiency_1d", "pmm_efficiency_1d_jax",
           "pmm_efficiency_1d_segments",
           "pmm_jones_1d", "pmm_jones_1d_segments", "PMMStack",
           "pmm_efficiency_1d_slanted", "pmm_jones_1d_slanted",
           "pmm_jones_1d_slanted_segments", "pmm_1d",
           "grating_convergence_class", "classify_from_grating"]

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
    ia = {int(o): i for i, o in enumerate(oa)}
    ib = {int(o): i for i, o in enumerate(ob)}
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
    """Indices of the largest group of PASSIVE solves that mutually agree
    PER-ORDER (and on the Jones matrix) within ``tol`` -- the converged plateau.

    ``records[i]`` is the ``rec`` tuple for :func:`_aligned_max_diff`; ``passive``
    is the aligned bool list.  Returns the index list (sorted) when it reaches
    ``min_cluster`` members, else ``[]``.  Clustering on the per-order signature
    (not the total) is what rejects an under-converged-but-energy-passive solve:
    such a degree fails to agree with the higher-degree plateau and is excluded.
    """
    pidx = [i for i, p in enumerate(passive) if p]
    best = []
    for a in pidx:
        grp = [b for b in pidx if _aligned_max_diff(records[a], records[b]) <= tol]
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
    the shift vanishes.  ``robust`` forces the NOISE-ROBUST forward branch even at
    ``kx0 == 0`` (the legacy ``Im(q) >= 0`` branch is bit-identical to the prior
    binary solve but has dense isolated-degree resonances for many-element /
    multi-region cells; the robust branch suppresses them).  Binary passes
    ``robust=False`` (bit-identical); the segmented solver passes ``robust=True``.
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
    q2, Acoef = _safe_geig(A, B)
    q = np.sqrt(q2)
    if kx0 or robust:
        # NOISE-ROBUST forward branch: the operator is (near-)Hermitian for
        # lossless media, so the QZ eig leaks ~1e-15 imag noise; the naive sign
        # test would flip near-real (propagating) modes on noise -> dense
        # spurious resonances.  Flip only when CLEARLY backward.
        tol = 1e-8 * max(float(np.max(np.abs(q))), 1.0)
        flip = (q.imag < -tol) | ((np.abs(q.imag) <= tol) & (q.real < 0.0))
        q = np.where(flip, -q, q)
    else:
        q = np.where(q.imag < 0.0, -q, q)           # Im(q) >= 0 forward decay
    lam = -1j * q
    return Acoef, lam, q, invop


def _sem_fourier_projection(orders, period, mats):
    """``T[m, i] = (1/period) INT phi_i(x) exp(-i m G x) dx`` for the global
    nodal Lagrange basis ``phi_i``; exact per element by oversampled Gauss
    quadrature (the integrand is oscillatory)."""
    from numpy.polynomial.legendre import leggauss
    l2g, elem_bnds, degree = mats["l2g"], mats["elem_bnds"], mats["degree"]
    ref_nodes, n_glob = mats["ref_nodes"], mats["n_glob"]
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


def _redheffer_star(SA, SB):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=_C)
    D = np.linalg.inv(I - B11 @ A22)
    F = np.linalg.inv(I - A22 @ B11)
    return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
            B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)


def _kz_forward(eps, kx):
    """``kz/k0`` on the forward branch for ``exp(-i w t)``: ``Im(kz) >= 0`` so
    the forward wave ``exp(+i kz z)`` decays."""
    val = np.sqrt(np.asarray(eps - kx ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


def _assemble_jones_farfield(Hsup, Hsub, S11, S21, orders, kx,
                             kz_sup, kz_sub, kz_inc, kx0n, N):
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
    """
    safe_r = np.where(np.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = np.where(np.abs(kz_sub) < 1e-12, 1.0, kz_sub)
    if abs(kz_inc) < 1e-9:
        raise ValueError(
            "pmm: grazing/evanescent incidence (kz_inc ~ 0) -- the incident wave "
            "carries ~no z-flux, so the R/T flux normalization is ill-defined.  "
            "Reduce the incidence angle below grazing.")
    m0 = np.where(orders == 0)[0][0]
    jones = np.zeros((2, 2), dtype=_C)
    R_eff = np.zeros((2, N))
    T_eff = np.zeros((2, N))
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
        R_eff[col] = np.where(np.real(kz_sup) > 1e-12, np.real(Re), 0.0)
        T_eff[col] = np.where(np.real(kz_sub) > 1e-12, np.real(Te), 0.0)
        jones[0, col] = rx[m0]                  # PUBLIC convention -> no conjugation
        jones[1, col] = ry[m0]
    return R_eff, T_eff, jones


def _scalar_farfield_RT(r_ord, t_ord, kx, kx0, k0, eps_sup, eps_sub,
                        polarization):
    """TE/TM diffraction efficiencies from the order-resolved reflection /
    transmission amplitudes of a scalar PMM solve.

    The TE channel normalizes by ``kz`` (the standard plane-wave z-flux); the TM
    channel normalizes by the wall-normal flux ``kz/eps`` (the inverse-rule
    channel), with the incident flux ``kz_inc/eps_sup``.  Orders below cut-off
    (``Re(kz) <= 0``) carry zero efficiency.  Shared verbatim by the vertical
    scalar core and the scalar slant solver."""
    kz_sup = _kz_forward(eps_sup, kx)
    kz_sub = _kz_forward(eps_sub, kx)
    kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
    if abs(kz_inc) < 1e-9:
        raise ValueError(
            "pmm: grazing/evanescent incidence (kz_inc ~ 0) -- the incident wave "
            "carries ~no z-flux, so R/T normalization is ill-defined.  Reduce the "
            "incidence angle below grazing.")
    if polarization == "te":
        R = np.real(kz_sup / kz_inc) * np.abs(r_ord) ** 2
        T = np.real(kz_sub / kz_inc) * np.abs(t_ord) ** 2
    else:
        flux_inc = np.real(kz_inc / eps_sup)
        R = np.real(kz_sup / eps_sup) * np.abs(r_ord) ** 2 / flux_inc
        T = np.real(kz_sub / eps_sub) * np.abs(t_ord) ** 2 / flux_inc
    R = np.where(np.real(kz_sup) > 1e-12, np.real(R), 0.0)
    T = np.where(np.real(kz_sub) > 1e-12, np.real(T), 0.0)
    return R, T


# ===========================================================================
# Core single-polarization PMM solve
# ===========================================================================

def _n_propagating_orders(period, wl, n_max):
    """Highest propagating Rayleigh order |m| with |kx_m| < n_max*k0."""
    return int(np.floor(float(np.real(n_max)) * period / wl + 1e-9))


def _pmm_solve(period, n_ridge, n_groove, n_sub, n_sup, depth, duty, wl,
               degree, polarization, n_ridge_el, n_groove_el, grade,
               far_field_orders, angle=0.0):
    """Binary (ridge/groove) single-degree scalar PMM solve -- a thin wrapper
    that builds the two-region operators and defers to :func:`_pmm_solve_core`
    (shared with the multi-region :func:`_pmm_solve_segments`)."""
    eps_ridge, eps_groove = n_ridge ** 2, n_groove ** 2
    eps_sup, eps_sub = n_sup ** 2, n_sub ** 2
    k0 = 2.0 * np.pi / wl
    d_wall = duty * period
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
    (``robust`` forces the noise-robust forward branch -- see :func:`_sem_modes`)."""
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

    Wsup, _ls, q_sup, invsup = _sem_modes(mats_sup, k0, polarization, kx0, robust)
    Wsub, _lb, q_sub, invsub = _sem_modes(mats_sub, k0, polarization, kx0, robust)
    if polarization == "te":
        Vsup, Vsub = Wsup @ np.diag(q_sup), Wsub @ np.diag(q_sub)
    else:
        Vsup = (invsup @ Wsup) @ np.diag(q_sup)
        Vsub = (invsub @ Wsub) @ np.diag(q_sub)
    # (region half-spaces do not propagate, so no region lam is needed.)

    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
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
                               polarization)
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


def _sem_modes_tensor(mats, k0, kx0=0.0, robust=False):
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

    Mbig = np.block([[G @ Cxx, G @ Cxy],
                     [Cyx,     Cyy - Kx2]])
    q2, W2 = np.linalg.eig(Mbig)
    q = np.sqrt(q2)
    # modal magnetic partner: V = Q @ W @ diag(1/lam) with the (Ky=0) Q block
    Q = np.block([[Cyx, Cyy - Kx2], [-Cxx, -Cxy]])

    if not kx0 and not robust:
        q = np.where(q.imag < 0.0, -q, q)   # Im(q) >= 0 forward decay (legacy)
    else:
        # POYNTING-FLUX forward selector: V2 partner is [Hx; Hy] and the modal H
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
        fscale = 1e-9 * max(float(np.max(np.abs(flux))), 1.0)
        prop = np.abs(flux) > fscale
        flip = np.where(prop, flux < 0.0, q.imag < 0.0)
        q = np.where(flip, -q, q)
    lam = -1j * q
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    V2 = Q @ W2 @ np.diag(1.0 / safe)
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
    Wsup, Vsup, _ls, _qs = _sem_modes_tensor(mats_sup, k0, kx0, robust)
    Wsub, Vsub, _lb, _qb = _sem_modes_tensor(mats_sub, k0, kx0, robust)

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
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    def _project(Wmodes):
        """Project each (Ex / Ey) half of the nodal modes onto the Rayleigh
        orders -> a (2N, modes) operator: x-orders stacked over y-orders."""
        return np.vstack([Tp @ Wmodes[:n_glob, :], Tp @ Wmodes[n_glob:, :]])

    Hsup = _project(Wsup)
    Hsub = _project(Wsub)

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
def _stabilize_scalar(solve_at_degree, d0, label):
    """Per-order convergence consensus over a degree window; ``solve_at_degree(d)
    -> (orders, R, T)``.  Shared by the binary + segmented scalar solvers."""
    scanned = []
    for d in range(d0, d0 + _STABILIZE_MAX_SCAN):
        orders, R, T = solve_at_degree(d)
        tot = float(np.real(R.sum() + T.sum()))
        scanned.append((d, orders, R, T, tot <= 1.0 + _PASSIVE_TOL))
        records = [(s[1], (s[2], s[3]), None) for s in scanned]
        passive = [s[4] for s in scanned]
        cluster = _converged_cluster(records, passive, _PER_ORDER_TOL,
                                     _MIN_PLATEAU)
        if not cluster:
            continue
        pick = 0 if 0 in cluster else cluster[0]
        return scanned[pick][1], scanned[pick][2], scanned[pick][3]
    passives = [s for s in scanned if s[4]]
    if not passives:
        raise RuntimeError(
            f"{label}: no resonance-free solve in degrees "
            f"[{d0}, {d0 + _STABILIZE_MAX_SCAN}); the requested degree sits in a "
            f"high-degree resonance band.  Use a lower degree (PMM converges "
            f"spectrally -- degree<=32 typically suffices) or "
            f"elements_per_region>1 with grade=True.")
    warnings.warn(
        f"{label}: the per-order solution did not converge within degrees "
        f"[{d0}, {d0 + _STABILIZE_MAX_SCAN}); returning the highest degree tried "
        f"(degree {max(p[0] for p in passives)}).  It is likely UNDER-RESOLVED "
        f"(the total power can be passive while the per-order efficiencies are "
        f"still wrong) -- raise degree or use elements_per_region>1 with "
        f"grade=True.", stacklevel=3)
    best = max(passives, key=lambda s: s[0])
    return best[1], best[2], best[3]


def _stabilize_jones(solve_at_degree, d0, label):
    """Per-order + Jones convergence consensus; ``solve_at_degree(d) ->
    (orders, R, T, J)``.  Shared by the binary + segmented anisotropic solvers."""
    scanned = []
    for d in range(d0, d0 + _STABILIZE_MAX_SCAN):
        o, R, T, J = solve_at_degree(d)
        tot = float(np.real(R.sum() + T.sum()))
        scanned.append((d, o, R, T, J, tot <= 2.0 + 2.0 * _JONES_PASSIVE_TOL))
        records = [(s[1], (s[2], s[3]), s[4]) for s in scanned]
        passive = [s[5] for s in scanned]
        cluster = _converged_cluster(records, passive, _PER_ORDER_TOL,
                                     _MIN_PLATEAU)
        if not cluster:
            continue
        pick = 0 if 0 in cluster else cluster[0]
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


def pmm_jones_1d(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic 1-D binary grating by the Polynomial Modal Method -- the full
    complex ``2x2`` Jones reflection (the spectral-element counterpart of
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d`).

    The ridge and groove are full ``(3, 3)`` permittivity tensors -- IN-PLANE OR
    OUT-OF-PLANE (an off-diagonal-z ``exz/eyz/ezx/ezy`` cell routes to the metric
    generator); the off-diagonal ``exy`` couples ``E_x`` and ``E_y`` in the
    spectral-element modal eigenproblem, so the response is a full Jones matrix (the
    phase relationship the scalar :func:`pmm_efficiency_1d` cannot carry).  Converges SPECTRALLY in the polynomial ``degree`` with no
    accuracy floor -- the PMM win on metals where the FMM needs many orders and
    the ASR stretch plateaus.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        FULL ``(3, 3)`` permittivity tensors of the ridge / groove (PUBLIC
        convention ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an
        isotropic region; build LC tensors with
        :func:`~lumenairy.elements.rcwa.uniaxial_tensor`.  OUT-OF-PLANE coupling
        (``eps_xz / eps_yz / eps_zx / eps_zy != 0``, e.g. a tilted-director LC or
        a magneto-optic medium) is supported via the native full-3x3 metric
        generator (the ``eps_zz``-Schur of ``E_z`` is folded pointwise, Li 1999
        Eq.12); per-order matches :func:`~lumenairy.elements.rcwa.rcwa_jones_1d`
        to ``<1e-3``.  In-plane tensors take the (byte-identical) in-plane path.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices (PUBLIC ``n =
        n + i kappa``).
    depth, duty_cycle, wavelength : float
        As in :func:`pmm_efficiency_1d` (the ridge occupies ``duty_cycle`` of
        the period).
    angle : float, optional
        Incidence angle (radians) in the x-z plane (classical mount, ``ky=0``).
        Oblique is supported via the ``+i kx0`` Bloch shift; the coupled tensor
        modes' forward set is chosen by the z-Poynting flux.  Lossless / mild-
        loss anisotropic (the tunable-LC case) is robust across angle; very
        lossy metal-corner TM at steep angle can be resonance-limited.
    degree : int, optional
        Polynomial degree per spectral element -- the spectral convergence knob.
        Default 16.
    elements_per_region : int, optional
        Spectral elements per homogeneous subsection (ridge / groove).  Default
        1.  Raise with ``grade=True`` to resolve the wall-corner field.
    grade : bool, optional
        Cluster the elements toward the walls when
        ``elements_per_region > 1``.  Default ``True``.
    far_field_orders : int, optional
        Rayleigh order count for the once-only forward far-field projection
        (auto-grown to cover the propagating orders).  Default 21.
    stabilize : bool, optional
        Guard against the isolated-degree PMM resonances (a near-singular
        layer<->region mode-match injects spurious flux and inflates
        ``sum(R)+sum(T)``).  When ``True`` (default) the solver scans a short
        upward degree window and returns the lowest degree at/above the request
        whose BOTH incident-polarization totals are energy-passive.  Set
        ``False`` to solve at exactly ``degree``.

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices (the far-field projection set).
    R_eff, T_eff : (2, M) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is the
        response to an incident ``E_x`` wave, row 1 to incident ``E_y`` (cross-
        polarization included).
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis (PUBLIC
        ``exp(-i w t)`` convention); columns are the responses to incident
        ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.  Matches
        :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` to the convergence
        tolerance.

    Notes
    -----
    NumPy / SciPy (dense generalized eig) by default.  **JAX-differentiable** for an
    IN-PLANE tensor (``exz = eyz = ezx = ezy = 0``) on a VERTICAL grating when any
    index / geometry / tensor argument is a JAX array: it routes to a self-contained
    ``jax.numpy`` twin (the 2n x 2n coupled ``[E_x; E_y]`` standard eig solved
    directly by the reused ``rcwa`` custom-VJP eig), returning ``jax.grad``-able
    ``R_eff`` / ``T_eff`` and the 2x2 ``jones`` w.r.t. the tensor entries (incl. the
    off-diagonal ``exy`` / ``eyx`` cross-pol coupling, real and imaginary), ``depth``,
    ``wavelength``, ``angle`` and the half-space indices.  The numpy path is
    **byte-identical** (the JAX branch fires only on JAX inputs).  ``d/d(angle)`` at
    EXACTLY normal incidence is a symmetry-protected zero where the degenerate TE/TM
    eigen-subspace leaves a tiny (~5e-5) artifact against the true zero -- harmless
    (the gradient there IS zero; no gauge fix is applied since it would corrupt the
    gauge-invariant ``|J|`` observables); differentiate at any oblique angle for a
    clean value.  ``stabilize=True``, ``elements_per_region>1``, a SLANTED wall, and
    an OUT-OF-PLANE tensor raise on the JAX path (NumPy-only).  Normal or oblique
    incidence, binary grating, full ``(3, 3)`` tensor (in-plane OR out-of-plane).  An
    out-of-plane tensor routes through the native full-3x3 metric generator (VERTICAL
    only).  Out-of-plane combined with a slanted wall is SUPPORTED (binary) as of
    2026-06-07: :func:`pmm_jones_1d_slanted` carries the slant as EXACT convection
    (``tan * d/dx``) instead of the static metric fold (whose ``ezz*tan^2`` re-
    injection capped the per-order accuracy at ~1e-2 for the full tensor), so
    out-of-plane + slant reaches the same ~1e-4 wall-normal per-order floor as the
    in-plane slant TM channel (validated vs an RCWA tensor z-staircase across slant
    15-60 deg, normal + oblique).  The multi-region (segments) out-of-plane + slant
    path remains guarded.
    """
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d: degree must be >= 2.")
    # A TRACED (jit) duty_cycle has no concrete value to range-check.
    if not is_jax_array(duty_cycle):
        if not (0.0 < float(duty_cycle) < 1.0):
            raise ValueError(
                f"pmm_jones_1d: duty_cycle must be strictly in (0, 1), got {duty_cycle}.")

    # JAX (differentiable) dispatch -- mirror pmm_efficiency_1d / rcwa.  Routed
    # to the self-contained jnp Jones twin ONLY when a tensor / geometry / angle
    # input is a JAX array; NumPy inputs fall through to the original
    # (byte-identical) code.  The JAX surface is binary, IN-PLANE tensor
    # (exz/eyz/ezx/ezy == 0), VERTICAL wall, NORMAL or OBLIQUE incidence, a
    # single fixed degree.  This branch MUST precede the np.asarray(_C) coercion
    # below, which would sever the JAX trace.  is_jax_array on a (3,3) tensor
    # array detects a JAX tensor input.
    _jax_inputs = (eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
                   wavelength, duty_cycle, angle)
    if any(is_jax_array(a) for a in _jax_inputs):
        if stabilize:
            raise ValueError(
                "pmm_jones_1d: the JAX (differentiable) path requires "
                "stabilize=False (the degree-scan returns a discrete degree via "
                "host control flow, which is non-differentiable); pass a single "
                "fixed degree where the numpy solve already conserves energy.")
        import jax.numpy as _jnp
        er_j = _jnp.asarray(eps_ridge, _jnp.complex128)
        eg_j = _jnp.asarray(eps_groove, _jnp.complex128)
        if er_j.shape[-2:] != (3, 3) or eg_j.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d: eps_ridge / eps_groove must be (3, 3) "
                "permittivity tensors (use scalar * np.eye(3) for isotropic).")
        # OUT-OF-PLANE (exz/eyz/ezx/ezy != 0) and SLANT stay NumPy-only (the
        # native metric generator is not differentiable yet) -- a precise raise
        # so the user is never silently given a wrong path.  Use the CONCRETE
        # off-plane magnitude when available (a traced off-plane entry cannot be
        # checked, so an all-traced tensor is assumed in-plane and validated by
        # the numpy oracle in tests).
        def _off_mag(M):
            try:
                arr = np.asarray(M, dtype=_C)
                return float(np.max(np.abs(arr[[0, 1, 2, 2], [2, 2, 0, 1]])))
            except Exception:
                return 0.0
        scale = 1.0
        try:
            scale = max(float(np.max(np.abs(np.asarray(eps_ridge, dtype=_C)))),
                        float(np.max(np.abs(np.asarray(eps_groove, dtype=_C)))),
                        1.0)
        except Exception:
            scale = 1.0
        off = max(_off_mag(eps_ridge), _off_mag(eps_groove))
        if off > 1e-9 * scale:
            raise NotImplementedError(
                "pmm_jones_1d: the JAX (differentiable) path supports IN-PLANE "
                "tensors only (exz/eyz/ezx/ezy == 0); the out-of-plane native "
                "metric generator is NumPy-only.")
        return _pmm_jones_1d_jax(
            period, er_j, eg_j, n_substrate, n_superstrate, depth, duty_cycle,
            wavelength, angle=angle, degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders))

    er = np.asarray(eps_ridge, dtype=_C)
    eg = np.asarray(eps_groove, dtype=_C)
    if er.shape[-2:] != (3, 3) or eg.shape[-2:] != (3, 3):
        raise ValueError(
            "pmm_jones_1d: eps_ridge / eps_groove must be (3, 3) permittivity "
            "tensors (use scalar * np.eye(3) for an isotropic region).")
    # FULL-3x3 OUT-OF-PLANE (exz/eyz/ezx/ezy != 0): route to the native full-3x3
    # metric generator (_pmm_jones_slant_solve at slant=0 -- the same operator the
    # slant/oblique paths use, now carrying the out-of-plane pointwise ezz-Schur).
    # The in-plane tensor keeps the (byte-identical) second-order _pmm_jones_solve.
    # Normal AND oblique are supported; per-order matches rcwa_jones_1d to <1e-3.
    scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    off = max(float(np.max(np.abs(er[[0, 1, 2, 2], [2, 2, 0, 1]]))),
              float(np.max(np.abs(eg[[0, 1, 2, 2], [2, 2, 0, 1]]))))
    if off > 1e-9 * scale:
        sargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, 0.0)
        skw = dict(n_ridge_el=int(elements_per_region),
                   n_groove_el=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_slant_solve(*sargs, degree=int(degree),
                                                   **skw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_solve(*sargs, degree=d, **skw)[:4],
            int(degree), "pmm_jones_1d")

    args = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
            duty_cycle, wavelength)
    kw = dict(n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_solve(*args, degree=int(degree), **kw)
        return o, R, T, J
    # Per-order + Jones convergence consensus (rejects the super-unity resonances
    # AND the under-resolved-but-energy-passive degrees; see _stabilize_jones).
    return _stabilize_jones(
        lambda d: _pmm_jones_solve(*args, degree=d, **kw)[:4], int(degree),
        "pmm_jones_1d")


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
    tol = 1e-8 * jnp.maximum(jnp.max(jnp.abs(q)), 1.0)
    flip = (q.imag < -tol) | ((jnp.abs(q.imag) <= tol) & (q.real < 0.0))
    q = jnp.where(flip, -q, q)
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
    # the scalar TM normaliser is flux_inc = kz_inc/eps_sup (mirror of
    # _pmm_solve_core -- the Ez longitudinal flux is already folded into the
    # scalar 1/eps weighting, unlike the Jones path's explicit (1+(kx0/kz)^2)).
    kx0n = kx0 / k0
    kz_inc = jnp.real(_kzf(eps_sup, jnp.asarray(kx0n, cj)))
    if polarization == "te":
        R = jnp.real(kz_sup / kz_inc) * jnp.abs(r_ord) ** 2
        T = jnp.real(kz_sub / kz_inc) * jnp.abs(t_ord) ** 2
    else:
        flux_inc = jnp.real(kz_inc / eps_sup)
        R = jnp.real(kz_sup / eps_sup) * jnp.abs(r_ord) ** 2 / flux_inc
        T = jnp.real(kz_sub / eps_sub) * jnp.abs(t_ord) ** 2 / flux_inc
    R = jnp.where(jnp.real(kz_sup) > 1e-12, jnp.real(R), 0.0)
    T = jnp.where(jnp.real(kz_sub) > 1e-12, jnp.real(T), 0.0)
    return orders, R, T


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
    ``d/d(wl)`` are valid only BETWEEN Rayleigh-order cutoffs."""
    import jax.numpy as jnp

    from .rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("pmm_efficiency_1d")

    if int(elements_per_region) != 1:
        raise NotImplementedError(
            "pmm_efficiency_1d: the JAX (differentiable) path currently "
            "supports elements_per_region == 1 only (the binary de-risking "
            "spike); use the NumPy path for hp-refinement.")

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
    wl_c = _re_or_none(wavelength)
    if wl_c is None:
        # wavelength is the traced grad variable: size from the far_field_orders
        # floor only (m_prop unknown without a concrete wl), capped at n_glob.
        wl_c = float("inf")
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
    fscale = 1e-9 * jnp.maximum(jnp.max(jnp.abs(flux)), 1.0)
    prop = jnp.abs(flux) > fscale
    flip = jnp.where(prop, flux < 0.0, q.imag < 0.0)
    q = jnp.where(flip, -q, q)
    lam = -1j * q
    safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
    V2 = Q @ W2 @ jnp.diag(1.0 / safe)
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
    Wsup, Vsup, _ls, _qs = _jpmm_sem_modes_tensor(mats_sup, jnp, eig, k0, kx0)
    Wsub, Vsub, _lb, _qb = _jpmm_sem_modes_tensor(mats_sub, jnp, eig, k0, kx0)

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
        rows_R.append(jnp.where(jnp.real(kz_sup) > 1e-12, jnp.real(Re), 0.0))
        rows_T.append(jnp.where(jnp.real(kz_sub) > 1e-12, jnp.real(Te), 0.0))
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
    (stabilize, multi-region, out-of-plane, slant) raises (handled upstream)."""
    import jax.numpy as jnp

    from .rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("pmm_jones_1d")

    if int(elements_per_region) != 1:
        raise NotImplementedError(
            "pmm_jones_1d: the JAX (differentiable) path currently supports "
            "elements_per_region == 1 only; use the NumPy path for "
            "hp-refinement.")

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
    wl_c = _re_or_none(wavelength)
    if wl_c is None:
        wl_c = float("inf")
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


def pmm_efficiency_1d(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D binary grating by the
    Polynomial Modal Method (subsectional spectral element; Edee 2011).

    A NON-Fourier alternative to :func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`
    that converges SPECTRALLY in the polynomial ``degree`` (no Gibbs at the
    walls), with no accuracy floor and well-conditioned interfaces -- see the
    module docstring for why this beats the Fourier method and the ASR
    coordinate stretch for metals / high-contrast TM.

    Parameters
    ----------
    period, n_ridge, n_groove, n_substrate, n_superstrate, depth, duty_cycle,
    wavelength : as in :func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`
        (metres / PUBLIC ``n = n + i kappa``).  The ridge occupies the fraction
        ``duty_cycle`` of the period.
    angle : float, optional
        Incidence angle (radians).  Oblique is supported via the ``+i kx0``
        Bloch shift of the pseudo-periodic envelope (the convection term is
        antisymmetrized so the wall-varying ``1/eps`` weight is handled
        correctly for TM); the forward modes use a noise-robust branch.
        Dielectric is robust across angle; very lossy metal-corner TM at steep
        angle can be resonance-limited (``stabilize`` may raise -- use rcwa).
    polarization : {'te', 'tm'}, optional
        ``'te'`` (E along the grooves) or ``'tm'``.  Default ``'te'``.
    degree : int, optional
        Polynomial degree per spectral element -- the SPECTRAL convergence knob
        (raise it for accuracy, not the element count).  Default 16.
    elements_per_region : int, optional
        Spectral elements per homogeneous subsection (ridge / groove).  Default
        1.  Raise (e.g. 2-4) with ``grade=True`` to resolve the metal-corner
        field singularity -- the speed lever for TM (hp-refinement).
    grade : bool, optional
        When ``elements_per_region > 1``, cluster the elements toward the walls
        (Chebyshev-Lobatto) to resolve the corner singularity.  Default
        ``True`` (no effect for ``elements_per_region == 1``).
    far_field_orders : int, optional
        Rayleigh order count for the once-only forward far-field projection
        (auto-grown to cover the propagating orders; kept well below the nodal
        DOF).  Default 21.
    stabilize : bool, optional
        Guard against the measure-zero discrete resonances at isolated
        polynomial degrees (a near-singular layer<->region mode-match injects
        spurious flux and inflates ``sum(R)+sum(T)``; the analogue of the FMM
        ``stabilize`` flag).  When ``True`` (default) the solver scans a short
        UPWARD degree window and returns the CONSENSUS result -- the value the
        converged degrees agree on (discarding both super-unity resonances and
        low-degree under-convergence), not merely the minimum or maximum power --
        build-reproducible and never below the requested degree's accuracy.  Set
        ``False`` to solve at exactly ``degree`` (e.g. for
        convergence studies that tolerate the occasional resonant degree).

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices (the far-field projection set).
    R_eff, T_eff : (M,) float ndarray
        Reflected / transmitted diffraction efficiency per order (real power
        fractions; evanescent orders 0).  For a lossless grating
        ``sum(R)+sum(T) == 1``; with loss the deficit is absorptance.

    Notes
    -----
    NumPy / SciPy (dense generalized eig) by default.  **JAX-differentiable** when
    any index / geometry argument is a JAX array (mirrors ``rcwa``): the call then
    routes to a self-contained ``jax.numpy`` twin and returns ``jax.grad``-able
    efficiencies w.r.t. ``eps`` (via ``n``, including COMPLEX/lossy eps -- both the
    real and imaginary parts, ``holomorphic=False``), ``depth``, ``wavelength``,
    ``angle`` and ``n_superstrate`` (the Bloch ``kx0`` convection is traced), AND
    ``duty_cycle`` (the moving grating wall -- a smooth fixed-topology shape
    gradient: the wall sits on an element boundary so the element Jacobians and the
    Rayleigh-projection phases carry it analytically, no remeshing), reusing the
    validated Lorentzian-broadened custom-VJP eig from ``rcwa``.  The numpy
    path is **byte-identical** (the JAX branch fires only on JAX inputs).  The
    differentiable surface is binary, NORMAL or OBLIQUE incidence, lossless or
    lossy eps, ``elements_per_region=1``, a single fixed ``degree`` with
    ``stabilize=False`` (a resonant degree gives the converged jnp answer but a
    non-physical numpy one -- pick a degree where the numpy solve already conserves
    energy); ``stabilize=True`` and ``elements_per_region>1`` raise on the JAX path
    (NumPy-only for now).  For a LOSSY cell, validate per-order / absorbed-fraction
    against an oracle, NOT energy (a lossless cell auto-balances power even with a
    wrong split).  Gradients are valid BETWEEN Rayleigh-order cutoffs (the
    propagating-order count is fixed per trace; set ``far_field_orders`` high enough
    to cover the working wavelength when differentiating ``wavelength``).  Requires
    ``jax_enable_x64`` (a warning fires otherwise -- complex64 is too ill-conditioned
    for the modal eig).  TM converges monotone-with-no-floor but only
    spectral-*ish* (the discontinuous TM partner is C0-averaged at the wall);
    ``elements_per_region>1, grade=True`` recovers the rate for metals.
    """
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d: polarization must be 'te' or 'tm', got "
            f"{polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d: degree must be >= 2.")
    # A TRACED (jit) duty_cycle has no concrete value to range-check; the JAX
    # path enforces 0 < duty < 1 on the concrete proxy / leading value instead.
    if not is_jax_array(duty_cycle):
        if not (0.0 < float(duty_cycle) < 1.0):
            raise ValueError(
                f"pmm_efficiency_1d: duty_cycle must be strictly in (0, 1), got "
                f"{duty_cycle}.")

    # JAX (differentiable) dispatch -- mirror rcwa's backend detection.  Routed
    # to the self-contained jnp twin ONLY when an index / geometry input is a
    # JAX array; NumPy inputs fall through to the original code verbatim (so the
    # NumPy path stays BYTE-IDENTICAL).  The JAX path covers the minimal
    # validated surface: binary, NORMAL incidence, fixed single degree.  This
    # branch MUST precede the _C(...) coercion below, which would sever the JAX
    # trace.
    if any(is_jax_array(a) for a in (n_ridge, n_groove, n_substrate,
                                     n_superstrate, depth, wavelength,
                                     duty_cycle, angle)):
        if stabilize:
            raise ValueError(
                "pmm_efficiency_1d: the JAX (differentiable) path requires "
                "stabilize=False (the degree-scan returns a discrete degree via "
                "host control flow, which is non-differentiable); pass a single "
                "fixed degree.")
        return _pmm_efficiency_1d_jax(
            period, n_ridge, n_groove, n_substrate, n_superstrate, depth,
            duty_cycle, wavelength, angle=angle, polarization=pol,
            degree=int(degree), elements_per_region=int(elements_per_region),
            grade=bool(grade), far_field_orders=int(far_field_orders))

    args = (period, _C(n_ridge), _C(n_groove), _C(n_substrate),
            _C(n_superstrate), depth, duty_cycle, wavelength)
    kw = dict(polarization=pol, n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        orders, R, T, _ = _pmm_solve(*args, degree=int(degree), **kw)
        return orders, R, T
    # Robust degree selection by per-order CONVERGENCE CONSENSUS: collect the
    # PASSIVE solves (total within _PASSIVE_TOL of unity -- discards the
    # super-unity resonances) and lock onto the plateau the converged degrees
    # AGREE ON per-order (the total alone is conserved even when under-resolved);
    # return the requested degree if it is in the plateau, else the lowest
    # converged degree, else warn/raise.  See _stabilize_scalar.
    return _stabilize_scalar(
        lambda d: _pmm_solve(*args, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d")


def pmm_efficiency_1d_jax(
    period,
    n_ridge,
    n_groove,
    n_substrate,
    n_superstrate,
    depth,
    duty_cycle,
    wavelength,
    *,
    angle=0.0,
    polarization="te",
    degree=16,
    elements_per_region=1,
    grade=True,
    far_field_orders=21,
):
    """JAX (differentiable) twin of :func:`pmm_efficiency_1d`.

    Thin wrapper that promotes its inputs to ``jax.numpy`` arrays and forwards to
    the unified :func:`pmm_efficiency_1d`, which auto-dispatches to the
    differentiable JAX backend on JAX inputs.  Prefer calling
    ``pmm_efficiency_1d(...)`` with ``jax.numpy`` arguments directly; this explicit
    twin exists for discoverability and the cross-backend parity contract.  The JAX
    path uses ``stabilize=False`` (passed automatically) -- choose a single fixed
    ``degree`` where the NumPy solve already conserves energy.  Differentiable
    w.r.t. ``n_ridge`` / ``n_groove`` (via eps), ``depth``, ``wavelength``,
    ``angle`` and the half-space indices; first-order only (no eig Hessian).
    """
    from ..backend import JAX_AVAILABLE as _JAX_AVAILABLE
    if not _JAX_AVAILABLE:
        raise ImportError(
            "pmm_efficiency_1d_jax requires the optional 'jax' extra; install with "
            "`pip install lumenairy[jax]` (or `pip install jax`).  Use the NumPy "
            "pmm_efficiency_1d for non-differentiable evaluation.")
    import jax.numpy as jnp
    return pmm_efficiency_1d(
        period,
        jnp.asarray(n_ridge),
        jnp.asarray(n_groove),
        jnp.asarray(n_substrate),
        jnp.asarray(n_superstrate),
        jnp.asarray(depth),
        duty_cycle,
        jnp.asarray(wavelength),
        angle=jnp.asarray(angle),
        polarization=polarization,
        degree=degree,
        elements_per_region=elements_per_region,
        grade=grade,
        far_field_orders=far_field_orders,
        stabilize=False,
    )


def pmm_efficiency_1d_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar diffraction efficiencies of an ARBITRARY piecewise-constant 1-D
    grating by the PMM -- the multi-region / multi-level generalization of
    :func:`pmm_efficiency_1d` (the 2-region ridge/groove special case).

    The PMM's fast isotropic path: each region carries a scalar (possibly
    complex) refractive index, a region wall lands on every spectral-element
    boundary (so ``eps`` is exact per element -- no Gibbs), and the solve
    converges spectrally in ``degree`` with no accuracy floor.  For anisotropic
    (tensor) regions use :func:`pmm_jones_1d_segments`.

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_efficiency_1d`.
    segments : list of (width_fraction, n)
        Consecutive regions along ``x`` (the ridge side first), each a
        ``(width_fraction, refractive_index)`` pair; the fractions must sum to 1
        (within ``1e-6``).  Covers multi-level staircases (blazed-grating
        approximations) and arbitrary multi-region cells.
    angle, polarization, degree, elements_per_region, grade, far_field_orders,
    stabilize : as in :func:`pmm_efficiency_1d`.

    Returns
    -------
    orders, R_eff, T_eff : as in :func:`pmm_efficiency_1d`.
    """
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d_segments: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError(
            "pmm_efficiency_1d_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    seg_n = [_C(n) for _, n in segments]
    sa = (period, widths, seg_n, _C(n_substrate), _C(n_superstrate), depth,
          wavelength)
    kw = dict(polarization=pol, n_el_per_region=int(elements_per_region),
              grade=bool(grade), far_field_orders=int(far_field_orders),
              angle=float(angle))

    if not stabilize:
        o, R, T, _ = _pmm_solve_segments(*sa, degree=int(degree), **kw)
        return o, R, T
    return _stabilize_scalar(
        lambda d: _pmm_solve_segments(*sa, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d_segments")


def pmm_jones_1d_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic 1-D grating with an ARBITRARY piecewise-constant profile by
    the PMM -- the multi-region / multi-level generalization of
    :func:`pmm_jones_1d` (the 2-segment ridge/groove special case) and the
    spectral-element counterpart of
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d_segments`.

    Each region carries its own (possibly anisotropic) IN-PLANE permittivity, so
    the response is a full complex ``2x2`` Jones reflection.  Covers multi-level
    staircases, interdigitated / N-region cells, and mixed isotropic / liquid-
    crystal regions (e.g. the grounded-tooth | LC | floating-tooth | LC device
    class).  Converges spectrally in ``degree`` with no accuracy floor.

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    segments : list of (width_fraction, eps)
        Consecutive regions along ``x``; each ``eps`` is a scalar (isotropic
        region) or a ``(3, 3)`` IN-PLANE permittivity tensor.  Width fractions
        must sum to 1 (within ``1e-6``).  Accepts the output of the
        ``grating_segments`` / ``binary_grating_segments`` /
        ``interdigitated_grating_segments`` builders.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize : as
        in :func:`pmm_jones_1d`.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d`.
    """
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError("pmm_jones_1d_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    tensors = []
    for _w, eps in segments:
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:                         # scalar -> isotropic tensor
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d_segments: each segment eps must be a scalar or a "
                "(3, 3) permittivity tensor.")
        tensors.append(M)
    # FULL-3x3 OUT-OF-PLANE (exz/eyz/ezx/ezy != 0): route to the native full-3x3
    # metric generator (_pmm_jones_slant_segments_solve at slant=0 -- region-count-
    # agnostic, carrying the out-of-plane pointwise ezz-Schur).  The in-plane
    # tensor keeps the (byte-identical) second-order _pmm_jones_solve_segments.
    scale = max([float(np.max(np.abs(M))) for M in tensors] + [1.0])
    off = max(float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]])))
              for M in tensors)
    if off > 1e-9 * scale:
        oargs = (period, widths, tensors, _C(n_substrate), _C(n_superstrate),
                 depth, wavelength, 0.0)
        okw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_slant_segments_solve(
                *oargs, degree=int(degree), **okw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_segments_solve(*oargs, degree=d, **okw)[:4],
            int(degree), "pmm_jones_1d_segments")
    sa = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
          wavelength)
    kw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_solve_segments(*sa, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_solve_segments(*sa, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_segments")


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


def _pmm_union_grid(layer_segments):
    """Build the shared nodal grid for a stack: the union of every layer's walls.

    ``layer_segments[i]`` = layer ``i``'s ``[(width_fraction, eps), ...]``.
    Returns ``(union_widths, layer_eps_union)`` where ``union_widths`` are the
    fractional widths of the union cells (sum to 1) and ``layer_eps_union[i][c]``
    is layer ``i``'s permittivity on union cell ``c`` (each union cell lies wholly
    within one of layer ``i``'s segments, so eps is exact per cell)."""
    walls = {0.0, 1.0}
    cums = []
    for segs in layer_segments:
        w = np.asarray([float(s[0]) for s in segs], dtype=float)
        cw = np.concatenate([[0.0], np.cumsum(w)])
        cw[-1] = 1.0
        cums.append(cw)
        walls.update(float(x) for x in cw)
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
        for w in uwalls[1:]:
            if w - keep[-1] > tol:
                keep.append(w)
        if keep[-1] < uwalls[-1]:           # never drop the period boundary
            keep[-1] = uwalls[-1]
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


class PMMStack:
    """Multilayer 1-D grating stack solved by the Polynomial Modal Method -- the
    spectral-element counterpart of :class:`~lumenairy.elements.rcwa.RCWAStack`.

    Compose anisotropic (or isotropic) 1-D patterned layers and uniform spacers
    between a superstrate and substrate, set the incident plane wave, and solve
    once for the diffraction efficiencies of both incident polarizations plus the
    zeroth-order ``2x2`` Jones reflection.  The whole stack is solved on the
    UNION of every layer's walls (one shared nodal grid), so each layer converges
    spectrally in ``degree`` with no Fourier truncation in-plane.

    Example
    -------
    >>> st = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
    >>> st.add_layer(0.2e-6, eps=2.1)                       # uniform spacer
    >>> st.add_layer(0.3e-6, segments=[(0.5, lc), (0.5, 1.0)])  # patterned
    >>> orders, R, T, jones = st.set_source(0.55e-6, angle=0.2).solve()

    Parameters
    ----------
    period : float
        Grating period (metres).
    n_substrate, n_superstrate : complex, optional
        Transmission / incidence half-space indices.
    degree : int, optional
        Polynomial degree per spectral element (the spectral knob).  Default 16.
    elements_per_region, grade, far_field_orders : as in
        :func:`pmm_jones_1d_segments`.
    factorization : {'auto', 'convection', 'covariant'}, optional
        Slant treatment (as in :func:`pmm_jones_1d_slanted`).  ``'auto'``
        (default) uses the SPECTRAL covariant oblique-coordinate generator for a
        stack whose layers share a single non-zero slant -- IN-PLANE OR
        OUT-OF-PLANE (the full-3x3 coupling enters via the Li Eq.12 ezz-Schur
        composites + cos*Dop cross blocks) -- and the (algebraic-but-fully-
        general) convection generator otherwise -- vertical or MIXED-slant stacks
        (the covariant oblique frame is per-slant, so a mixed-slant cascade falls
        back to convection).  ``'covariant'`` forces the spectral path (raises on
        mixed/zero slant); ``'convection'`` forces the general path.

    Notes
    -----
    Anisotropic / Jones throughout (scalar layers are promoted to isotropic
    tensors), FULL ``(3,3)`` tensors IN-PLANE OR OUT-OF-PLANE
    (``eps_xz/yz/zx/zy``), normal or oblique incidence, NumPy (not JAX).  Layers
    may be VERTICAL or SLANTED (``add_layer(..., slant_angle=...)``), in-plane or
    out-of-plane, and freely mixed: an all-vertical-in-plane stack uses the
    symmetric ``+/-q`` cascade (bit-identical to the prior release), and any
    slanted OR out-of-plane layer promotes the whole stack to the general
    forward/backward S-matrix (solved by the div-conforming metric generator,
    which carries the slant fold AND the out-of-plane ezz-Schur).  A slanted
    out-of-plane layer reaches the same ~1e-4 wall-normal per-order floor as the
    single-layer solver (validated vs an RCWA tensor z-staircase; energy
    conserves).  The modal forward set uses the z-Poynting-flux
    selector (as the multi-region single-layer solver), so the many-element shared
    grid stays resonance-free.
    """

    def __init__(self, period, *, n_substrate=1.0, n_superstrate=1.0,
                 degree=16, elements_per_region=1, grade=True,
                 far_field_orders=21, factorization="auto"):
        if int(degree) < 2:
            raise ValueError("PMMStack: degree must be >= 2.")
        if factorization not in ("auto", "convection", "covariant"):
            raise ValueError(
                "PMMStack: factorization must be 'auto', 'convection' or "
                f"'covariant', got {factorization!r}.")
        self.period = float(period)
        self.n_sub = _C(n_substrate)
        self.n_sup = _C(n_superstrate)
        self.degree = int(degree)
        self.n_el = int(elements_per_region)
        self.grade = bool(grade)
        self.factorization = factorization
        self.ffo = int(far_field_orders)
        self._layers = []                          # (thickness, segments)
        self._src = None

    def _as_tensor(self, eps):
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "PMMStack.add_layer: each eps must be a scalar or a (3, 3) "
                "permittivity tensor.")
        return M

    @staticmethod
    def _is_oop(M):
        """True if the (3,3) tensor has out-of-plane coupling (eps_xz/yz/zx/zy)."""
        scale = max(float(np.max(np.abs(M))), 1.0)
        return float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]]))) > 1e-9 * scale

    def add_layer(self, thickness, *, segments=None, eps=None, slant_angle=0.0):
        """Append a layer.  Give exactly one of ``eps`` (uniform: scalar or
        ``(3,3)`` tensor) or ``segments`` (a list of ``(width_fraction, eps)`` --
        each ``eps`` scalar or ``(3,3)``; widths sum to 1).  ``slant_angle``
        (radians) tilts the layer's straight side-walls from the vertical (``0``
        = vertical); a slanted layer is solved by the div-conforming covariant-
        metric generator and cascaded via the general fwd/back S-matrix, so a
        stack may MIX vertical and slanted layers.  Returns ``self``."""
        if (segments is None) == (eps is None):
            raise ValueError(
                "PMMStack.add_layer: give exactly one of `segments` or `eps`.")
        if eps is not None:
            segs = [(1.0, self._as_tensor(eps))]
        else:
            if len(segments) < 1:
                raise ValueError("PMMStack.add_layer: empty segments.")
            segs = [(float(w), self._as_tensor(e)) for w, e in segments]
        self._layers.append((float(thickness), segs, float(slant_angle)))
        return self

    def set_source(self, wavelength, *, angle=0.0):
        """Set the incident plane wave (vacuum wavelength [m], incidence
        ``angle`` [rad] in the x-z plane).  Returns ``self``."""
        self._src = dict(wl=float(wavelength), angle=float(angle))
        return self

    def solve(self):
        """Solve the stack.  Returns ``(orders, R_eff, T_eff, jones_reflection)``
        as :func:`pmm_jones_1d_segments` (``R_eff`` / ``T_eff`` are ``(2, M)``:
        row 0 = incident ``E_x``, row 1 = incident ``E_y``; ``jones`` is the
        zeroth-order ``2x2`` reflection)."""
        if self._src is None:
            raise ValueError("PMMStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("PMMStack.solve: add at least one layer.")
        wl, angle = self._src["wl"], self._src["angle"]
        k0 = 2.0 * np.pi / wl
        kx0 = float(np.real(self.n_sup)) * np.sin(angle) * k0
        eps_sup, eps_sub = self.n_sup ** 2, self.n_sub ** 2

        # ---- factorization dispatch: covariant (SPECTRAL slant) vs convection --
        # 'auto' uses the covariant oblique-coordinate generator (spectral TM) for
        # ANY slanted stack -- in-plane OR out-of-plane (the full-3x3 coupling
        # enters via the Li Eq.12 ezz-Schur composites + cos*Dop cross blocks) --
        # and the convection metric generator otherwise (all-vertical).  The
        # covariant cascade still requires a UNIFORM slant.  'covariant' forces it
        # (raises on
        # out-of-plane); 'convection' forces the algebraic-but-fully-general path.
        _oop = any(self._is_oop(M) for L in self._layers for _w, M in L[1])
        _slants = [abs(L[2]) for L in self._layers]
        _signed = [L[2] for L in self._layers]
        # The covariant oblique frame is per-slant (the shear u = x - tanφ z), so
        # the spectral covariant cascade requires a UNIFORM slant across all layers
        # (and the homogeneous half-spaces are solved in that same frame).  A
        # MIXED-slant stack (e.g. a vertical spacer + a slanted grating) would need
        # inter-layer lateral-shift corrections and falls back to convection.
        # Gate uniformity on the SIGNED slant (not abs): the shear u = x - tanφ z is
        # MIRROR-sheared for +φ vs -φ, so an equal-magnitude opposite-sign stack
        # would cascade -φ layer modes against half-spaces fixed in the +φ frame --
        # incompatible gauges, a silently-wrong S-matrix.  Opposite-sign / mixed
        # slants fall back to convection ('auto') or raise ('covariant').
        _uniform_slant = (max(_slants) >= _COV_MIN_SLANT_RAD
                          and (max(_signed) - min(_signed)) <= 1e-12)
        _fac = self.factorization
        if _fac == "auto":
            _fac = "covariant" if _uniform_slant else "convection"
        if _fac == "covariant":
            if not _uniform_slant:
                raise NotImplementedError(
                    "PMMStack: factorization='covariant' requires a UNIFORM "
                    "non-zero slant across all layers (the covariant oblique "
                    "frame is per-slant); use 'convection' or 'auto' for "
                    "vertical / mixed-slant stacks.")
            return self._solve_covariant(wl, angle, k0)

        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers])
        nU = len(uwidths)
        layer_mats = [
            _build_sem_tensor_segments(
                self.period, uwidths, [_tensor3_dict(e) for e in eps_u],
                self.degree, self.n_el, self.grade)
            for eps_u in layer_eps_u]
        t_sup = _tensor3_dict(eps_sup * np.eye(3))
        t_sub = _tensor3_dict(eps_sub * np.eye(3))
        mats_sup = _build_sem_tensor_segments(
            self.period, uwidths, [t_sup] * nU, self.degree, self.n_el, self.grade)
        mats_sub = _build_sem_tensor_segments(
            self.period, uwidths, [t_sub] * nU, self.degree, self.n_el, self.grade)
        n_glob = mats_sup["n_glob"]

        Wsup, Vsup, _l, _g = _sem_modes_tensor(mats_sup, k0, kx0, True)
        Wsub, Vsub, _l, _g = _sem_modes_tensor(mats_sub, k0, kx0, True)

        # Redheffer recursion: sup -> [interface, propagation]*layers -> sub.
        # A layer needs the GENERAL fwd/back cascade if it is SLANTED or carries
        # full-3x3 OUT-OF-PLANE coupling (both break the +/-q symmetry and are
        # solved by the metric generator); an all-vertical-in-plane stack keeps the
        # symmetric path (bit-identical to the prior release).
        oop_layer = [any(self._is_oop(M) for _w, M in L[1]) for L in self._layers]
        if not any(abs(L[2]) > 1e-12 or oo
                   for L, oo in zip(self._layers, oop_layer)):
            # ALL-VERTICAL IN-PLANE: the symmetric (+/-q) S-matrix path --
            # bit-identical to the prior release.
            lmodes = [_sem_modes_tensor(m, k0, kx0, True) for m in layer_mats]
            S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
            for i, (Wl, Vl, lam_l, _q) in enumerate(lmodes):
                S = _redheffer_star(S, _propagation_smatrix(
                    lam_l, k0 * self._layers[i][0]))
                nW, nV = ((Wsub, Vsub) if i == len(lmodes) - 1
                          else (lmodes[i + 1][0], lmodes[i + 1][1]))
                S = _redheffer_star(S, _interface_smatrix(Wl, Vl, nW, nV))
        else:
            # MIXED vertical / SLANTED / OUT-OF-PLANE: every layer carries EXPLICIT
            # forward/backward modes cascaded with the GENERAL fwd/back S-matrix.  A
            # SLANTED or OUT-OF-PLANE layer uses the div-conforming metric generator
            # (_layer_modes_metric carries both the slant fold AND the out-of-plane
            # pointwise ezz-Schur); a plain vertical-in-plane layer reuses its
            # symmetric modes as the degenerate [[W,W],[V,-V]] forward/backward set.
            Ms = _half_M_sym_metric(Wsup, Vsup)
            Mb = _half_M_sym_metric(Wsub, Vsub)
            Mls, lamfs, lambs = [], [], []
            for i, (_thk, _segs, slant) in enumerate(self._layers):
                if abs(slant) > 1e-12 or oop_layer[i]:
                    mm = _build_nodal_metric_segments(
                        self.period, uwidths,
                        [_tensor3_dict(e) for e in layer_eps_u[i]],
                        self.degree, self.n_el, self.grade)
                    Wf, Vf, lamf, _qf, Wb, Vb, lamb, _qb = _layer_modes_metric(
                        mm, k0, slant, kx0)
                else:
                    Wl, Vl, lam_l, _q = _sem_modes_tensor(
                        layer_mats[i], k0, kx0, True)
                    Wf, Wb, Vf, Vb = Wl, Wl, Vl, -Vl
                    lamf, lamb = lam_l, -lam_l
                Mls.append(np.block([[Wf, Wb], [Vf, Vb]]))
                lamfs.append(lamf)
                lambs.append(lamb)
            S = _interface_smatrix_general(Ms, Mls[0])
            for i in range(len(self._layers)):
                S = _redheffer_star(S, _propagation_smatrix_general(
                    lamfs[i], lambs[i], k0 * self._layers[i][0]))
                nextM = (Mb if i == len(self._layers) - 1 else Mls[i + 1])
                S = _redheffer_star(S, _interface_smatrix_general(Mls[i], nextM))
        S11, _S12, S21, _S22 = S

        # far-field projection (mirrors _pmm_jones_solve_core)
        # max over BOTH in-plane diagonal components -- TM sees exx, TE sees eyy, so
        # exx alone under-resolves a high-eyy stack and can miss a propagating order
        # (audit P2; consistent with every single-layer solver).
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
        m_prop = _n_propagating_orders(self.period, wl, n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        if 2 * m_prop + 1 > n_proj:               # parity with the single-layer cores
            raise ValueError(
                f"PMMStack.solve: degree={self.degree} too low to resolve the "
                f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
                f"degree or elements_per_region.")
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        G = 2.0 * np.pi / self.period
        kx = (kx0 + orders * G) / k0
        N = len(orders)
        Tp = _sem_fourier_projection(orders, self.period, mats_sup)

        def _proj(Wm):
            return np.vstack([Tp @ Wm[:n_glob, :], Tp @ Wm[n_glob:, :]])
        Hsup, Hsub = _proj(Wsup), _proj(Wsub)
        kz_sup = _kz_forward(eps_sup, kx)
        kz_sub = _kz_forward(eps_sub, kx)
        kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
        kx0n = kx0 / k0
        R_eff, T_eff, jones = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc, kx0n, N)
        return orders, R_eff, T_eff, jones

    def _solve_covariant(self, wl, angle, k0):
        """SPECTRAL multi-layer solve via the Li covariant oblique-coordinate
        generator (in-plane OR out-of-plane).  Parallels the general fwd/back
        cascade of
        :meth:`solve` but uses :func:`_cov_layer_4n` modes + COVARIANT
        homogeneous half-spaces on the shared union grid, so slanted layers
        converge SPECTRALLY (vertical-grade) instead of the convection
        generator's algebraic ~1e-4 floor.  Internally exp(+iwt): eps are
        conjugated in and the Jones conjugated out; the union widths/tensors are
        PRE-REVERSED to cancel the _segment_elem_bnds [::-1] (so orders land in
        the user's input frame, matching the convection path)."""
        period, degree, n_el, grade = self.period, self.degree, self.n_el, \
            self.grade
        kx0 = float(np.real(np.conj(self.n_sup))) * np.sin(angle) * k0
        eps_sup = np.conj(self.n_sup ** 2)
        eps_sub = np.conj(self.n_sub ** 2)
        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers])
        uw = list(uwidths)[::-1]
        nU = len(uwidths)

        def grid(eps_list):
            seg = [_t3_slant(np.conj(np.asarray(e, _C))) for e in eps_list][::-1]
            return _build_nodal_metric_segments(period, uw, seg, degree, n_el,
                                                grade)
        # the homogeneous half-spaces are solved in the SAME oblique frame as the
        # layers (uniform slant), so their modes share the layer convention; pass
        # PUBLIC eps (grid conjugates to the internal exp(+iwt) convention).
        slant = self._layers[0][2]
        layer_mats = [grid(layer_eps_u[i]) for i in range(len(self._layers))]
        mats_s = grid([self.n_sup ** 2 * np.eye(3) for _ in range(nU)])
        mats_b = grid([self.n_sub ** 2 * np.eye(3) for _ in range(nU)])
        n_glob = mats_s["n_glob"]

        # DIV-CONFORMING Ez closure if ANY layer is out-of-plane (machine-precision
        # OOP), applied uniformly to all layers + both half-spaces so the closure
        # matches across every interface; in-plane stacks keep the modal closure.
        divconf = any(self._is_oop(M) for _L in self._layers for _w, M in _L[1])
        Ws, Vs, kzs, fws = _cov_layer_4n(mats_s, k0, slant, kx0, divconf)
        Wb, Vb, kzb, fwb = _cov_layer_4n(mats_b, k0, slant, kx0, divconf)
        fs, _bs = _cov_split(Ws, Vs, kzs, fws)
        fb, _bb = _cov_split(Wb, Vb, kzb, fwb)

        def _msym(W, V, f):
            return np.block([[W[:, f], W[:, f]], [V[:, f], -V[:, f]]])
        Mls, lamf_l, lamb_l = [], [], []
        for i, (_thk, _segs, slant) in enumerate(self._layers):
            Wl, Vl, kzl, fwl = _cov_layer_4n(layer_mats[i], k0, slant, kx0,
                                             divconf)
            fl, bl = _cov_split(Wl, Vl, kzl, fwl)
            Mls.append(np.block([[Wl[:, fl], Wl[:, bl]],
                                 [Vl[:, fl], Vl[:, bl]]]))
            lamf_l.append(-1j * kzl[fl])
            lamb_l.append(-1j * kzl[bl])
        S = _interface_smatrix_general(_msym(Ws, Vs, fs), Mls[0])
        for i in range(len(self._layers)):
            S = _redheffer_star(S, _propagation_smatrix_general(
                lamf_l[i], lamb_l[i], k0 * self._layers[i][0]))
            nextM = (_msym(Wb, Vb, fb) if i == len(self._layers) - 1
                     else Mls[i + 1])
            S = _redheffer_star(S, _interface_smatrix_general(Mls[i], nextM))
        S11, _S12, S21, _S22 = S

        # max over BOTH in-plane diagonal components -- TM sees exx, TE sees eyy, so
        # exx alone under-resolves a high-eyy stack and can miss a propagating order
        # (audit P2; consistent with every single-layer solver).
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
        m_prop = _n_propagating_orders(period, wl, n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        if 2 * m_prop + 1 > n_proj:               # parity with the single-layer cores
            raise ValueError(
                f"PMMStack.solve (covariant): degree={self.degree} too low to "
                f"resolve the {2 * m_prop + 1} propagating orders "
                f"(n_glob={n_glob}); raise degree or elements_per_region.")
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        N = len(orders)
        kx = kx0 / k0 + orders * (2.0 * np.pi / period) / k0
        Tp = _sem_fourier_projection(orders, period, mats_s)
        kz_sup = _kz_forward(eps_sup, kx)
        kz_sub = _kz_forward(eps_sub, kx)
        kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
        Hsup = np.vstack([Tp @ Ws[:n_glob, fs], Tp @ Ws[n_glob:, fs]])
        Hsub = np.vstack([Tp @ Wb[:n_glob, fb], Tp @ Wb[n_glob:, fb]])
        R, T, jones = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc,
            kx0 / k0, N)
        return orders, R, T, np.conj(jones)        # conj: bridge +iwt -> public


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
        # shift already lives in A1's Lop).
        q2, Acoef = sla.eig(A1, A2)
        q = np.sqrt(q2)
        if kx0:
            tol = 1e-8 * max(float(np.max(np.abs(q))), 1.0)
            flip = (q.imag < -tol) | ((np.abs(q.imag) <= tol) & (q.real < 0.0))
            q = np.where(flip, -q, q)
        else:
            q = np.where(q.imag < 0.0, -q, q)
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
        score = np.where(prop, Sz, q.imag)
        fidx = np.argsort(-score)[:n]
    bidx = np.array(sorted(set(range(len(q))) - set(fidx.tolist())))

    Wf, Vf, qf = phis[:, fidx], V[:, fidx], q[fidx]
    Wb, Vb, qb = phis[:, bidx], V[:, bidx], q[bidx]
    lam_f = -1j * qf                              # exp(-lam_f k0 z) decays fwd
    lam_b = -1j * qb
    return dict(symmetric=False, Wf=Wf, Vf=Vf, lam_f=lam_f, qf=qf,
                Wb=Wb, Vb=Vb, lam_b=lam_b, qb=qb, invop=invop,
                Dop=Dop, t=t, k0=k0, polarization=polarization)


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
    md_s = _sem_modes_slant(mats_sup, k0, polarization, 0.0, kx0)
    md_b = _sem_modes_slant(mats_sub, k0, polarization, 0.0, kx0)
    Ml, lamf_l, lamb_l, _Wf_l = _modes_M_slant(md_l)
    Ms, _lamf_s, _lamb_s, Wf_s = _modes_M_slant(md_s)
    Mb, _lamf_b, _lamb_b, Wf_b = _modes_M_slant(md_b)

    S = _interface_smatrix_general(Ms, Ml)
    S = _redheffer_star(S, _propagation_smatrix_general(lamf_l, lamb_l,
                                                        k0 * depth))
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
                               polarization)
    if return_coeffs:
        m0 = int(np.where(orders == 0)[0][0])
        return orders, R, T, n_glob, _C(r_ord[m0]), _C(t_ord[m0])
    return orders, R, T, n_glob


def pmm_efficiency_1d_slanted(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D SLANTED binary grating by the
    inclined-coordinate Polynomial Modal Method (Granet, Randriamihaja &
    Raniriharinosy, JOSA A 34:975, 2017).

    Extends :func:`pmm_efficiency_1d` to a grating whose straight side-walls are
    tilted by ``slant_angle`` from the vertical.  Where a Fourier method
    (:func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`) must STAIRCASE the
    slant into many laterally-shifted thin layers and converge in the slice
    count, PMM solves a SINGLE slanted layer exactly in the inclined coordinate
    ``u = x - tan(slant_angle) z`` -- no z-staircase -- reaching the converged
    efficiencies at a fraction of the cost (~30-70x fewer DOF*slices in the
    validated dielectric cases).

    Parameters
    ----------
    period ... wavelength : as in :func:`pmm_efficiency_1d` (metres / PUBLIC
        ``n = n + i kappa``).  The (projected) ridge occupies ``duty_cycle`` of
        the period; the planar layer interfaces are at ``z = 0`` and
        ``z = depth``.
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical (then this
        reduces bit-identically to :func:`pmm_efficiency_1d`).  Validated for
        ``0 <= slant_angle <= ~75 deg``; conditioning grows as
        ``sec^2(slant_angle)`` at steep tilt.
    angle : float, optional
        Incidence angle (radians).  COMBINED oblique incidence + nonzero slant IS
        supported: it is routed through the metric-generator Jones solver
        (:func:`pmm_jones_1d_slanted`) with an isotropic tensor and the requested
        scalar channel extracted (see Notes), per-order to ~2-3e-3 vs an RCWA
        staircase.  At normal incidence (or a vertical grating) the dedicated
        inclined-coordinate scalar solver is used.
    polarization, degree, elements_per_region, grade, far_field_orders,
    stabilize : as in :func:`pmm_efficiency_1d`.

    Returns
    -------
    orders, R_eff, T_eff : as in :func:`pmm_efficiency_1d`.

    Notes
    -----
    The slant injects a linear-in-``q`` convection term (the modal eigenproblem
    becomes quadratic, companion-linearized) and breaks the ``+/-q`` field
    symmetry, so the explicit forward/backward generalized S-matrix is used.
    The inclined coordinate is INTERNAL to the patterned layer; the homogeneous
    half-spaces and the Rayleigh far-field projection are in the lab ``(x, z)``
    frame.  TE matches a fine RCWA staircase to ~1e-5; TM self-converges and is
    the BETTER reference (RCWA-TM is Gibbs/slice-limited).

    SCOPE: binary (1 ridge + 1 groove), any slant, NORMAL or OBLIQUE incidence.
    The dedicated inclined-coordinate scalar eigenproblem is used at normal
    incidence (and for a vertical grating at any angle); for combined oblique
    incidence + nonzero slant -- where that scalar solver's ``kx0 <-> slant``
    convection cross-coupling is unresolved -- the call is delegated to the
    genuine Edee-Granet 2024 metric generator via :func:`pmm_jones_1d_slanted`
    (isotropic tensor ``n^2 I``, scalar channel extracted: TE = E along the
    grooves = Jones row 1, TM = row 0), which resolves the cross-coupling with the
    round-19 div-conforming ``E_z`` closure (per-order to ~2-3e-3 vs an RCWA
    staircase -- the wall-normal TM/p-pol inverse-rule limit plus the oblique-slant
    convection, not a coupling error; the in-plane wall-normal TM floor alone is much
    tighter, ~3e-5..1e-4 near degree 18-20, U-shaped).
    """
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d_slanted: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d_slanted: degree must be >= 2.")
    if not (0.0 < float(duty_cycle) < 1.0):
        raise ValueError(
            f"pmm_efficiency_1d_slanted: duty_cycle must be strictly in (0, 1), got "
            f"{duty_cycle}.")
    if abs(float(angle)) > 1e-12 and abs(float(slant_angle)) > 1e-12:
        # COMBINED OBLIQUE + SLANT.  The scalar inclined-frame solver's own
        # Bloch<->slant convection cross-term is unresolved (energy conserves but
        # the per-order split is wrong), so route through the genuine Edee-Granet
        # 2024 metric generator (pmm_jones_1d_slanted), whose round-19 div-
        # conforming E_z closure DOES handle combined oblique + slant (per-order
        # to ~2-3e-3 vs an RCWA staircase, degree-cleanly).  An isotropic region
        # is the diagonal tensor n^2 * I; the Jones response is then uncoupled, so
        # the scalar channel is a single row: TE (E along the grooves, E_y) is
        # Jones row 1, TM (E_x) is row 0.
        eye3 = np.eye(3)
        o, R_j, T_j, _ = pmm_jones_1d_slanted(
            period, (_C(n_ridge) ** 2) * eye3, (_C(n_groove) ** 2) * eye3,
            _C(n_substrate), _C(n_superstrate), depth, duty_cycle, wavelength,
            float(slant_angle), angle=float(angle), degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders), stabilize=bool(stabilize))
        row = 1 if pol == "te" else 0
        return o, R_j[row], T_j[row]

    args = (period, _C(n_ridge), _C(n_groove), _C(n_substrate),
            _C(n_superstrate), depth, duty_cycle, wavelength,
            float(slant_angle))
    kw = dict(angle=float(angle), polarization=pol,
              elements_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders))

    if not stabilize:
        orders, R, T, _ = _pmm_slant_solve(*args, degree=int(degree), **kw)
        return orders, R, T
    return _stabilize_scalar(
        lambda d: _pmm_slant_solve(*args, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d_slanted")


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
        L[0:n, 0:n] += k0 * (-Kx @ EZZi @ EZX_L)
        L[0:n, n:2 * n] += k0 * (-Kx @ EZZi @ EZY_L)
        L[2 * n:3 * n, 3 * n:4 * n] += k0 * (-EYZ_L @ EZZi @ Kx)
        L[3 * n:4 * n, 3 * n:4 * n] += k0 * (EXZ_L @ EZZi @ Kx)

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
    Ws, Vs, _ls, _qs = _sem_modes_tensor(mats_sup, k0, kx0, True)
    Wsub, Vsub, _lb, _qb2 = _sem_modes_tensor(mats_sub, k0, kx0, True)
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
    S = _redheffer_star(
        S, _propagation_smatrix_general(lamf_l, lamb_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix_general(Ml, Mb))
    S11, _S12, S21, _S22 = S

    def _proj_fwd(M):
        Wf = M[:2 * n_glob, :2 * n_glob]
        return np.vstack([Tp @ Wf[:n_glob, :], Tp @ Wf[n_glob:, :]])
    Hsup = _proj_fwd(Ms)
    Hsub = _proj_fwd(Mb)

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
# convergence at slant.  Validated (proto): same physical answer as convection
# (cross-validated two independent formulations + RCWA Fourier-converged oracle),
# but spectral vs algebraic -> ~1e-7 by degree ~24 where convection is ~1e-4
# (100-2400x fewer degrees for matched accuracy); energy conserves to ~1e-6
# across cells x slant 15-60 x normal/oblique.
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
        EZZi = _safe_inv(Op(lambda t: t["ezz"]))          # (eps^33)^-1
        EXZ = Op(lambda t: t["exz"])
        EZX = Op(lambda t: t["ezx"])
        EYZ = Op(lambda t: t["eyz"])
        EZY = Op(lambda t: t["ezy"])
        # conical-projected single x-derivative.  OBLIQUE incidence: the transverse
        # derivative on the periodic envelope is d/dx + i*kx0 (field ~ e^{i kx0 x}
        # u(x), Granet Eq.17), the SAME Bloch shift the convection generator injects
        # via Dopx (_build_generator_metric).  At kx0=0 this is bare Dop, so normal
        # incidence stays byte-identical.
        cD = cos * (Dop + 1j * kx0 * I)
        M[n:2 * n, n:2 * n] += (EXZ @ EZZi) @ cD          # Hy<-Hy (exz)
        M[3 * n:4 * n, 3 * n:4 * n] += -cD @ (EZZi @ EZX)  # Ex<-Ex (ezx)
        M[2 * n:3 * n, n:2 * n] += -(EYZ @ EZZi) @ cD     # Hx<-Hy (eyz); minus
        M[3 * n:4 * n, 0:n] += -cD @ (EZZi @ EZY)         # Ex<-Ey (ezy)
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
    """Forward/backward split, exactly half each (by Poynting / decay)."""
    n = W.shape[0] // 2
    half = W.shape[1] // 2
    if int(np.sum(fwd)) == half:
        fidx = np.where(fwd)[0]
    else:
        Ex, Ey, Hx, Hy = W[:n], W[n:], V[:n], V[n:]
        Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
        sc = np.where(np.abs(np.imag(kz)) < 1e-7 * max(float(np.max(np.abs(kz))),
                                                       1.0), Sz, np.imag(kz))
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
    S = _redheffer_star(S, _propagation_smatrix_general(
        -1j * kzl[fl], -1j * kzl[bl], k0 * depth))
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
        kz = np.sqrt(_C(eps) - kx ** 2 + 0j)
        return np.where(np.imag(kz) < 0, -kz, kz)
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
    _exx_eq_ezz = max(abs(er[0, 0] - er[2, 2]), abs(eg[0, 0] - eg[2, 2]))
    if _inplane_off > 1e-9 * _scale or _exx_eq_ezz > 1e-9 * _scale:
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


def pmm_jones_1d_slanted(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
    factorization: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SLANTED binary grating with full ``(3, 3)`` permittivity tensors (in-plane
    OR out-of-plane) -- the anisotropic-Jones counterpart of
    :func:`pmm_efficiency_1d_slanted`, by the Edee-Granet convection-metric
    spectral-element solver.

    Combines the tilted side-walls of :func:`pmm_efficiency_1d_slanted` with the
    coupled ``(E_x, E_y)`` Jones response of :func:`pmm_jones_1d`.  The tilt
    enters as an EXACT first-order convection (``tan * d/dx`` on the clean
    vertical metric generator -- this is what lets out-of-plane + slant reach the
    wall-normal floor; see :func:`_build_generator_metric`), and the modal
    eigenproblem is the TRUE first-order physical Maxwell
    generator ``-i k gamma psi = L psi`` (``psi = [E_x; E_y; iZ H_x; iZ H_y]``,
    LINEAR in ``gamma``): the magnetic field is read directly from the
    eigenvector, so the layer modes are flux-orthogonal by construction (the
    symplectic property a reshaped convection pencil lacks) and energy conserves
    to ~1e-13 across ``0-60`` deg for both real-symmetric and gyrotropic tensors.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        IN-PLANE permittivity tensors of the ridge / groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  As in :func:`pmm_jones_1d`: pass
        ``scalar * np.eye(3)`` for an isotropic region; ``exy`` / ``eyx`` couple
        ``E_x`` and ``E_y`` (real-symmetric for a tilted LC director, anti-
        Hermitian ``+/- i g`` for a gyrotropic / magneto-optic medium).  FULL
        out-of-plane coupling (``eps_xz / eps_yz / eps_zx / eps_zy != 0``, e.g. a
        tilted-director LC) is SUPPORTED (binary grating): it routes through the
        coupled metric generator and reaches the same ~1e-4 wall-normal per-order
        floor as the in-plane slant TM channel (validated vs an RCWA tensor
        z-staircase, slant 15-60 deg, normal + oblique).
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices (PUBLIC ``n =
        n + i kappa``).
    depth, duty_cycle, wavelength : float
        As in :func:`pmm_efficiency_1d_slanted` (the projected ridge occupies
        ``duty_cycle`` of the period; the planar interfaces are at ``z = 0`` and
        ``z = depth``).
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical (then this
        reduces to :func:`pmm_jones_1d`).  Validated for ``0 <= slant_angle <=
        ~60 deg`` (conditioning grows as ``sec^2`` at steep tilt).
    angle : float, optional
        Incidence angle (radians).  COMBINED oblique incidence + nonzero slant is
        SUPPORTED (the round-19 div-conforming ``E_z`` closure removes the
        Bloch-amplified Liu-2015 spurious null; ``kx0 = k0 Re(n_sup) sin(angle)``
        is wired through the generator, the half-spaces, and the lab Rayleigh far
        field): the metric generator conserves energy ~1e-13 and the per-order
        split matches an RCWA staircase to ~2-3e-3 at oblique+slant.  (The in-plane
        wall-normal TM/p-pol inverse-rule floor itself is far tighter: it converges
        to ~3e-5..1e-4 near degree 18-20, U-shaped, then mildly degrades past ~degree
        22 as the benign flux-null evanescent spurious sea grows with slant -- energy
        stays ~1e-13 throughout; TE is machine-clean ~1e-6.)  Combined oblique + slant
        always routes through the metric generator (the scalar diagonal cure's
        oblique+slant per-order is wrong).
    degree, elements_per_region, grade, far_field_orders, stabilize : as in
        :func:`pmm_jones_1d`.
    factorization : {'auto', 'convection', 'covariant'}, optional
        Slant treatment.  ``'auto'`` (default) picks the best path per cell:
        ``'covariant'`` for ANY slanted cell -- in-plane OR out-of-plane (the
        spectral win) -- and ``'convection'`` otherwise (vertical).
        ``'convection'`` carries the tilt as an exact first-order convection on
        the lab-Cartesian metric generator -- robust at all slants and tensors,
        but the TM/p-pol per-order accuracy converges ALGEBRAICALLY (~1e-4 at
        practical degree).  ``'covariant'`` routes through the Li-1999 oblique-
        coordinate covariant generator: the slanted wall becomes a coordinate
        surface, so the wall-normal discontinuity is handled algebraically and the
        TM channel converges SPECTRALLY (vertical-grade ~1e-7 by degree ~24) --
        the SAME physical answer as ``'convection'`` but ~100-2400x fewer degrees
        for matched accuracy.  ``'covariant'`` handles diagonal, coupled
        (``exy``/``eyx``), AND OUT-OF-PLANE (full 3x3 ``eps_xz/yz/zx/zy``)
        tensors, normal + oblique, lossless + lossy -- the out-of-plane coupling
        enters via the pointwise ezz-Schur composites (Li Eq.12) + the ``cos*Dop``
        single-derivative cross blocks (Li Eq.18/19), spectral at slant, so
        ``'auto'`` routes out-of-plane slanted cells to it too (~15x fewer degrees
        than convection).  Pass ``'convection'`` explicitly to force the fully-
        general
        algebraic path (e.g. for byte-stable cross-checks).

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices.
    R_eff, T_eff : (2, M) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is the
        response to an incident ``E_x`` wave, row 1 to incident ``E_y`` (cross-
        polarization included).
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis (PUBLIC
        ``exp(-i w t)`` convention); columns are the responses to incident
        ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.

    Notes
    -----
    NumPy / SciPy (dense eig); not JAX-differentiable.  SCOPE: BINARY grating
    (1 ridge + 1 groove), full ``(3, 3)`` tensor IN-PLANE OR OUT-OF-PLANE, normal
    OR oblique incidence at any slant.  The multi-region
    :func:`pmm_jones_1d_slanted_segments` companion carries full ``(3, 3)``
    tensors in-plane OR out-of-plane on every factorization.

    DIAGONAL CURE (round 16; Granet 2017 JOSA A 34:975 / Granet 2023; Liu 2015
    CiCP 18:467).  A DIAGONAL tensor (``exy = eyx = 0``) WITH ``exx == ezz`` in
    BOTH regions at NORMAL incidence (or any incidence on a VERTICAL grating) is
    solved through the DIV-CONFORMING scalar slant operator (``_sem_modes_slant``:
    the Li ``1/eps`` inverse rule sits INSIDE the z-stiffness, so it is free of the
    Liu-2015 spurious harmonic-mean static mode) -- TE via ``n = sqrt(eyy)``, TM
    via ``n = sqrt(exx)``, assembled into the diagonal Jones.  This is the FASTER
    (scalar ``n x n`` vs tensor ``4n x 4n``) and MORE-ACCURATE-per-degree path for
    that case.

    METRIC GENERATOR (round 11 + round-19 div-conforming closure).  COUPLED
    tensors (``exy / eyx != 0``), diagonal tensors with ``exx != ezz``, AND ALL
    combined oblique+slant cases use the convection-metric ``[E;H]`` first-order
    generator (``_build_generator_metric``).  As of round 19 its ``E_z``
    elimination is DIV-CONFORMING at ALL slants (``1/ezz`` placed BETWEEN the
    discrete z-derivatives, ``+ iS0 INT(1/ezz) B' B'``, Granet 2023 Eq.16-18 /
    Popov-Neviere App.B), so its TM-block spectrum bit-matches the scalar slant
    solver, is free of the Liu-2015 spurious harmonic-mean null, and per-order TM
    converges to the scalar oracle (~6.5e-4 deg32 / ~4.3e-4 deg40 at 45 deg).
    Energy conserves to ~1e-13 in all paths.  The combined oblique+slant far field
    (inclined-frame consistent, lab half-spaces) is degree-clean -- no stabilize
    crutch.
    """
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_slanted: degree must be >= 2.")
    if not (0.0 < float(duty_cycle) < 1.0):
        raise ValueError(
            f"pmm_jones_1d_slanted: duty_cycle must be strictly in (0, 1), got "
            f"{duty_cycle}.")
    er = np.asarray(eps_ridge, dtype=_C)
    eg = np.asarray(eps_groove, dtype=_C)
    if er.shape[-2:] != (3, 3) or eg.shape[-2:] != (3, 3):
        raise ValueError(
            "pmm_jones_1d_slanted: eps_ridge / eps_groove must be (3, 3) "
            "permittivity tensors (use scalar * np.eye(3) for an isotropic "
            "region).")
    # OUT-OF-PLANE (eps_xz/yz/zx/zy != 0): SUPPORTED (2026-06-07).  Since the
    # slant is carried as EXACT convection in the metric generator (see
    # _build_generator_metric), out-of-plane + slant now reaches the SAME
    # wall-normal ~1e-4 per-order floor as the in-plane slant TM channel
    # (validated vs an independent RCWA tensor z-staircase across slant 15-60 deg,
    # normal + oblique, symmetric / lossy / asymmetric tensors; energy conserves
    # to ~1e-13).  An out-of-plane tensor MUST route through the coupled metric
    # generator and NOT the scalar diagonal cure (which is the z-decoupled in-plane
    # subset and would silently drop the off-plane coupling), so `off` excludes it
    # from the diagonal cure below.  BINARY grating only -- the multi-region
    # (segments) out-of-plane + slant path stays guarded pending its own
    # per-order validation.
    scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    off = max(float(np.max(np.abs(er[[0, 1, 2, 2], [2, 2, 0, 1]]))),
              float(np.max(np.abs(eg[[0, 1, 2, 2], [2, 2, 0, 1]]))))

    # ---- COVARIANT OBLIQUE-COORDINATE path (SPECTRAL slant) ----------------
    # `factorization='covariant'` routes the slanted layer through the Li-1999
    # oblique-coordinate covariant generator instead of the convection generator.
    # The slanted wall becomes a coordinate surface, so the TM/p-pol channel
    # converges SPECTRALLY (vertical-grade ~1e-7 by degree ~24) instead of the
    # convection path's ALGEBRAIC ~1e-4 floor -- same physical answer, ~100-2400x
    # fewer degrees.  Handles diagonal AND coupled (exy/eyx) IN-PLANE tensors,
    # normal + oblique, lossless + lossy, IN-PLANE OR OUT-OF-PLANE.  The full-3x3
    # out-of-plane coupling (eps_xz/yz/zx/zy) enters the covariant generator via
    # the pointwise ezz-Schur composites (Li Eq.12) + the cos*Dop single-x-
    # derivative cross blocks (see _cov_blocks / _cov_generator_4n) -- SPECTRAL at
    # slant, same as the in-plane covariant.  The DEFAULT 'auto' picks covariant
    # for ANY slanted cell (the spectral win, now including out-of-plane) and
    # convection otherwise (vertical, where the vertical solver is already exact).
    if factorization not in ("auto", "convection", "covariant"):
        raise ValueError(
            "pmm_jones_1d_slanted: factorization must be 'auto', 'convection' "
            f"or 'covariant', got {factorization!r}.")
    if factorization == "auto":
        factorization = ("covariant"
                         if abs(float(slant_angle)) >= _COV_MIN_SLANT_RAD
                         else "convection")
    if (factorization == "covariant"
            and abs(float(slant_angle)) < _COV_MIN_SLANT_RAD):
        # The covariant oblique frame u = x - tan(phi) z DEGENERATES at ~zero slant:
        # the isotropic half-spaces' TE/TM modes become EXACTLY degenerate, the
        # interface mode-match goes near-singular, and the solve becomes BLAS-build-
        # dependent (it blew up on CI's OpenBLAS while passing on MKL).  The grating
        # is vertical there, so defer to the EXACT vertical Jones solver (handles
        # in-plane AND out-of-plane), which is well-conditioned and deterministic.
        # stabilize=True (NOT the slanted call's flag): the vertical Jones solver
        # has LAPACK-build-dependent resonances at a FIXED degree, so its robust
        # degree-scan is needed here -- and it matches the reduction tests'
        # pmm_jones_1d reference (which uses the default stabilize=True), making the
        # slant=0 result byte-deterministic across BLAS builds.
        return pmm_jones_1d(period, eps_ridge, eps_groove, n_substrate,
                            n_superstrate, depth, duty_cycle, wavelength,
                            angle=float(angle), degree=int(degree),
                            elements_per_region=int(elements_per_region),
                            grade=bool(grade),
                            far_field_orders=int(far_field_orders),
                            stabilize=True)
    if factorization == "covariant":
        cargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, float(slant_angle))
        ckw = dict(n_ridge_el=int(elements_per_region),
                   n_groove_el=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_oblique_solve(*cargs, degree=int(degree),
                                                     **ckw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_oblique_solve(*cargs, degree=d, **ckw)[:4],
            int(degree), "pmm_jones_1d_slanted")

    # ---- THE DIAGONAL CURE (round 16, Granet 2017/2023; Liu 2015) -----------
    # For a DIAGONAL in-plane tensor (exy = eyx = 0) WITH exx == ezz BOTH
    # regions, the TE / TM channels decouple and each maps onto the scalar slant
    # operator _sem_modes_slant -- which is DIV-CONFORMING (the Li 1/eps inverse
    # rule sits INSIDE the z-stiffness) and so SPURIOUS-MODE-FREE.  Route TE
    # through the scalar slant with n=sqrt(eyy) and TM with n=sqrt(exx), then
    # assemble the diagonal Jones.  This is the MORE-ACCURATE path: it sheds the
    # latent ~2e-4 per-order Liu-2015 spurious-harmonic-mean accuracy gap that
    # the pointwise metric-generator Ez-elimination (_build_metric_generator,
    # O_inv_ezz = iS0 @ [[1/ezz]]) carries (energy still conserves to ~1e-12).
    # Coupled tensors (exy/eyx != 0) AND diagonal tensors with exx != ezz fall
    # through to the metric generator (now div-conforming at all slants, so the
    # latent gap is gone there too).
    #
    # COMBINED OBLIQUE + SLANT is the ONE case the diagonal cure must NOT take:
    # the scalar slant operator's per-order split is wrong for oblique + nonzero
    # slant (energy still conserves -- a lossless cell auto-balances total power --
    # which is exactly why :func:`pmm_efficiency_1d_slanted` forbids that combo).
    # The round-19 div-conforming METRIC GENERATOR is the validated oblique+slant
    # path (per-order matches an RCWA staircase to ~2e-3, degree-clean), so route
    # combined oblique+slant through it even for a diagonal cell.  Oblique on a
    # VERTICAL grating (slant=0) and slant at NORMAL incidence are both still
    # handled correctly by the (more-accurate) scalar cure.
    inplane_off = max(abs(er[0, 1]), abs(er[1, 0]),
                      abs(eg[0, 1]), abs(eg[1, 0]))
    exx_eq_ezz = max(abs(er[0, 0] - er[2, 2]), abs(eg[0, 0] - eg[2, 2]))
    combined_oblique_slant = (abs(float(angle)) > 1e-12
                              and abs(float(slant_angle)) > 1e-12)
    diagonal_cure = (inplane_off <= 1e-9 * scale
                     and exx_eq_ezz <= 1e-9 * scale
                     and off <= 1e-9 * scale          # out-of-plane -> metric gen
                     and not combined_oblique_slant)
    if diagonal_cure:
        dargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, float(slant_angle))
        dkw = dict(elements_per_region=int(elements_per_region),
                   grade=bool(grade), far_field_orders=int(far_field_orders),
                   angle=float(angle))
        if not stabilize:
            return _pmm_jones_slant_diag_solve(*dargs, degree=int(degree),
                                               **dkw)
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_diag_solve(*dargs, degree=d, **dkw),
            int(degree), "pmm_jones_1d_slanted")

    args = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
            duty_cycle, wavelength, float(slant_angle))
    kw = dict(n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_slant_solve(*args, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_slant_solve(*args, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_slanted")


def pmm_jones_1d_slanted_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
    factorization: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SLANTED multi-region grating with full ``(3, 3)`` permittivity tensors
    (IN-PLANE OR OUT-OF-PLANE) -- the multi-region generalization of
    :func:`pmm_jones_1d_slanted` and the slanted counterpart of
    :func:`pmm_jones_1d_segments`.  The default ``factorization='auto'`` picks the
    spectral covariant path for ANY slanted cell -- in-plane OR out-of-plane (the
    full-3x3 coupling enters via the Li Eq.12 ezz-Schur composites + cos*Dop cross
    blocks) -- and the convection path otherwise (vertical).
    ``factorization='covariant'`` carries out-of-plane too; pass ``'convection'``
    to force the fully-general algebraic path.

    Each region carries its own (possibly anisotropic) tensor, and the straight
    side-walls are tilted by ``slant_angle`` from the vertical.  Solved by the
    same div-conforming convection-metric ``[E;H]`` generator as the binary
    :func:`pmm_jones_1d_slanted` -- the metric generator and the lab-frame far
    field are region-count-agnostic, so the N-region cell uses the identical
    (validated) operator + half-space machinery on an N-segment nodal grid.
    Energy conserves to ~1e-13 across ``0-60`` deg for asymmetric multi-region
    cells (including coupled / gyrotropic regions); combined oblique + slant is
    supported (the metric generator handles it).

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    segments : list of (width_fraction, eps)
        Consecutive regions along ``x``; each ``eps`` is a scalar (isotropic
        region) or a ``(3, 3)`` IN-PLANE permittivity tensor.  Width fractions
        must sum to 1 (within ``1e-6``).
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` reduces to
        :func:`pmm_jones_1d_segments`.  Validated ``0 <= slant_angle <= ~60 deg``.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize,
    factorization : as in :func:`pmm_jones_1d_slanted`.  ``factorization=
        'covariant'`` gives SPECTRAL TM convergence for the multi-region cell too
        (in-plane OR out-of-plane); it pre-reverses the region order so the
        covariant far field lands in the user's input frame.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d_slanted`.

    Notes
    -----
    BINARY-cell reductions are bit-identical to :func:`pmm_jones_1d_slanted` and
    the ``slant=0`` limit reduces to :func:`pmm_jones_1d_segments` (to the
    div-conforming discretization difference).  Full ``(3, 3)`` tensor IN-PLANE
    OR OUT-OF-PLANE (out-of-plane + slant reaches the ~1e-4 wall-normal floor,
    validated vs a multi-region RCWA tensor z-staircase); NumPy/SciPy (not JAX).
    """
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_slanted_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError(
            "pmm_jones_1d_slanted_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    tensors = []
    for _w, eps in segments:
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:                         # scalar -> isotropic tensor
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d_slanted_segments: each segment eps must be a "
                "scalar or a (3, 3) permittivity tensor.")
        tensors.append(M)
    # ---- COVARIANT OBLIQUE-COORDINATE path (SPECTRAL slant, opt-in) ---------
    # Multi-region generalization of the binary covariant path; spectral TM
    # convergence vs the convection path's algebraic floor (same answer).
    # IN-PLANE OR OUT-OF-PLANE (the full-3x3 coupling enters via the Li Eq.12
    # ezz-Schur composites + the cos*Dop single-derivative cross blocks).
    if factorization not in ("auto", "convection", "covariant"):
        raise ValueError(
            "pmm_jones_1d_slanted_segments: factorization must be 'auto', "
            f"'convection' or 'covariant', got {factorization!r}.")
    if factorization == "auto":
        factorization = ("covariant"
                         if abs(float(slant_angle)) >= _COV_MIN_SLANT_RAD
                         else "convection")
    if (factorization == "covariant"
            and abs(float(slant_angle)) < _COV_MIN_SLANT_RAD):
        # The covariant oblique frame degenerates at ~zero slant (the isotropic
        # half-spaces' TE/TM modes go exactly degenerate -> near-singular interface,
        # BLAS-build-dependent).  Defer to the EXACT vertical Jones segments solver,
        # which is well-conditioned and deterministic.
        # stabilize=True: the vertical solver's robust degree-scan handles its
        # fixed-degree LAPACK resonances and matches the reduction reference.
        return pmm_jones_1d_segments(
            period, segments, n_substrate, n_superstrate, depth, wavelength,
            angle=float(angle), degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders), stabilize=True)
    if factorization == "covariant":
        ca = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
              wavelength, float(slant_angle))
        ckw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_oblique_segments_solve(
                *ca, degree=int(degree), **ckw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_oblique_segments_solve(*ca, degree=d, **ckw)[:4],
            int(degree), "pmm_jones_1d_slanted_segments")
    # OUT-OF-PLANE (eps_xz/yz/zx/zy != 0): SUPPORTED (2026-06-07).  Multi-region
    # out-of-plane + slant rides the SAME exact-convection slant treatment as the
    # binary path (see _build_generator_metric), so it reaches the ~1e-4 wall-
    # normal per-order floor (validated vs a multi-region RCWA tensor z-staircase
    # oracle: 3-region dT ~3e-4 at slant 30; 2-segment == binary to ~1e-14; energy
    # conserves).  Segments route straight through the coupled metric generator
    # (no scalar diagonal cure here), so no out-of-plane dispatch guard is needed.
    sa = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
          wavelength, float(slant_angle))
    kw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_slant_segments_solve(
            *sa, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_slant_segments_solve(*sa, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_slanted_segments")


def pmm_1d(
    period: float,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    eps_ridge=None,
    eps_groove=None,
    duty_cycle: float = 0.5,
    segments=None,
    slant_angle: float = 0.0,
    angle: float = 0.0,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unified 1-D anisotropic-Jones PMM dispatcher -- one entry point that
    auto-routes to the right solver by geometry.

    Covers the whole 1-D Jones family in a single call:

    ====================  =====================  =====================================
    geometry              ``slant_angle == 0``   ``slant_angle != 0``
    ====================  =====================  =====================================
    BINARY                :func:`pmm_jones_1d`    :func:`pmm_jones_1d_slanted`
    (``eps_ridge`` +      (vertical)              (slanted)
    ``eps_groove`` +
    ``duty_cycle``)
    MULTI-REGION          :func:`pmm_jones_1d_    :func:`pmm_jones_1d_slanted_segments`
    (``segments``)        segments`
    ====================  =====================  =====================================

    Each ``eps`` may be a scalar (promoted to an isotropic tensor) or a full
    ``(3, 3)`` permittivity tensor (IN-PLANE or, for the VERTICAL cases,
    OUT-OF-PLANE ``eps_xz/eyz/ezx/ezy != 0``).  Normal or oblique incidence.
    (Combined out-of-plane + slant is SUPPORTED by the binary and segments slanted
    solvers as of 2026-06-07 -- the slant is carried as exact convection, reaching
    the ~1e-4 wall-normal per-order floor.)

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    eps_ridge, eps_groove, duty_cycle : the BINARY grating spec (give these OR
        ``segments``, not both).
    segments : list of ``(width_fraction, eps)`` -- the MULTI-REGION spec.
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize : as in
        :func:`pmm_jones_1d`.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d`.
    """
    has_binary = eps_ridge is not None or eps_groove is not None
    if (segments is None) == (not has_binary):
        raise ValueError(
            "pmm_1d: give EITHER `segments` OR `eps_ridge` + `eps_groove` "
            "(+ `duty_cycle`), not both / neither.")
    common = dict(angle=float(angle), degree=int(degree),
                  elements_per_region=int(elements_per_region),
                  grade=bool(grade), far_field_orders=int(far_field_orders),
                  stabilize=bool(stabilize))
    slanted = abs(float(slant_angle)) > 1e-12
    if segments is not None:
        if slanted:
            return pmm_jones_1d_slanted_segments(
                period, segments, n_substrate, n_superstrate, depth, wavelength,
                float(slant_angle), **common)
        return pmm_jones_1d_segments(
            period, segments, n_substrate, n_superstrate, depth, wavelength,
            **common)
    if eps_ridge is None or eps_groove is None:
        raise ValueError(
            "pmm_1d: the binary spec needs BOTH `eps_ridge` and `eps_groove`.")
    if slanted:
        return pmm_jones_1d_slanted(
            period, eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
            float(duty_cycle), wavelength, float(slant_angle), **common)
    return pmm_jones_1d(
        period, eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
        float(duty_cycle), wavelength, **common)


# ===========================================================================
# Convergence-class predictor for right-angle grating edges (Li-Granet 2011)
# ===========================================================================
def grating_convergence_class(eps_quadrants):
    """Classify modal-method convergence at a right-angle grating edge (Li &
    Granet, JOSA A 28:738, 2011).

    Predicts whether -- and how fast -- a modal method (FMM/RCWA, AMM, or PMM)
    converges at the right-angle corner where four regions of permittivity
    meet.  The governing field singularity is in the IN-PLANE electric field
    (the TM problem); TE (E parallel to the edge) is typically far better
    behaved.  This is a PURE O(1) diagnostic -- it does not solve the grating.

    Parameters
    ----------
    eps_quadrants : sequence of 4 complex
        The four permittivities meeting at the corner, IN ORDER AROUND THE
        VERTEX ``(eps1, eps2, eps3, eps4)``; ``eps1``/``eps3`` and
        ``eps2``/``eps4`` are the diagonal pairs.  PUBLIC convention
        ``Im(eps) > 0`` for loss.

    Returns
    -------
    dict
        ``type`` ('I' | 'II' | 'III' | 'degenerate'), ``tau`` (complex
        singularity exponent, Eq. 2), ``delta`` (Eq. 4), ``delta_prime``
        (Eq. 3), ``predicted_rate`` (``Re[tau]``, the algebraic decay exponent
        for Type I; else NaN), ``converges`` (bool), ``warning`` (str).

    Classification (lossless edges)
    -------------------------------
    * **Type I**  ``0 < Delta < 1`` (all-dielectric): REGULAR singularity;
      modal methods converge ALGEBRAICALLY at rate ``Re[tau]`` (slow as
      ``Re[tau] -> 0``; ``elements_per_region>1, grade=True`` recovers it).
    * **Type II** ``Delta < 0`` (LOSSLESS metal-dielectric): IRREGULAR
      singularity; NO method (FMM/AMM/PMM) converges.  Mitigate with metal
      loss, a rounded corner, or accept non-convergence.
    * **Type III** ``Delta > 1`` (requires a metal quadrant): no singularity,
      fast.  IMPOSSIBLE for an all-dielectric corner -- the squared numerator
      over a positive denominator forces ``Delta <= 1``.

    Loss caveat
    -----------
    A lossless metal corner is Type II; absorption (``Im(eps) > 0``) lifts the
    irregularity ASYMPTOTICALLY and is reported as (slowly) convergent.  This is
    NOT a hard switch -- a weakly lossy metal corner can still stall like
    Type II at practical truncation, so read ``converges=True`` for a
    near-lossless metal as "convergent in the limit," not "fast".

    References
    ----------
    L. Li & G. Granet, "Field singularities at lossless metal-dielectric
    right-angle edges and their ramifications to the numerical modeling of
    gratings," J. Opt. Soc. Am. A 28, 738-746 (2011).
    """
    e1, e2, e3, e4 = (_C(v) for v in eps_quadrants)
    loss_tol = 1e-9
    degen_tol = 1e-12

    has_loss = max(abs(e.imag) for e in (e1, e2, e3, e4)) > loss_tol

    f12, f23, f34, f41 = e1 + e2, e2 + e3, e3 + e4, e4 + e1
    num = (e1 * e3 - e2 * e4) ** 2
    den = f12 * f23 * f34 * f41
    scale = (abs(e1) + abs(e2) + abs(e3) + abs(e4)) ** 4 + 1e-30
    if abs(den) < degen_tol * scale or not np.isfinite(num / den):
        return dict(
            type="degenerate", tau=complex("nan"), delta=complex("nan"),
            delta_prime=complex("nan"), predicted_rate=float("nan"),
            converges=False,
            warning=("DEGENERATE EDGE: a (eps_i+eps_j) denominator factor "
                     "vanished (impedance-matched / resonant corner); "
                     "Delta_prime is ill-defined."))

    delta_prime = num / den
    delta = 1.0 - delta_prime
    tau = (2.0 / np.pi) * cmath.asin(cmath.sqrt(complex(delta_prime)))
    dr = delta.real

    if has_loss:
        # Loss lifts the lossless-metal irregularity (asymptotically).
        ctype = "I" if dr < 1.0 else "III"
        warning = ""
        if dr < 0.0:
            ctype = "I"
            warning = (
                "Lossy metal corner: the lossless Type-II irregularity is "
                "lifted by absorption, but a WEAKLY lossy metal can still "
                "stall at practical truncation; convergence may be SLOW (use "
                "mesh grading toward the wall).")
        return dict(type=ctype, tau=tau, delta=delta, delta_prime=delta_prime,
                    predicted_rate=float(abs(tau.real)), converges=True,
                    warning=warning)

    if dr < 0.0:
        return dict(
            type="II", tau=tau, delta=delta, delta_prime=delta_prime,
            predicted_rate=float("nan"), converges=False,
            warning=("TYPE II IRREGULAR SINGULARITY (lossless metal-dielectric "
                     "edge): NO modal method (FMM/AMM/PMM) converges. Add metal "
                     "loss, round the corner, or accept non-convergence -- do "
                     "NOT trust efficiencies from this edge."))
    if dr > 1.0:
        return dict(type="III", tau=tau, delta=delta, delta_prime=delta_prime,
                    predicted_rate=float("nan"), converges=True, warning="")

    rate = tau.real
    warning = ""
    if rate < 0.3:
        warning = (
            f"WEAK Type-I singularity (Re[tau]={rate:.4f} < 0.3): algebraic "
            "convergence is SLOW (~N^-tau); use elements_per_region>1, "
            "grade=True to recover the rate.")
    return dict(type="I", tau=tau, delta=delta, delta_prime=delta_prime,
                predicted_rate=float(rate), converges=True, warning=warning)


def classify_from_grating(eps_superstrate, eps_ridge, eps_groove,
                          eps_substrate):
    """Convenience wrapper of :func:`grating_convergence_class` for a 1-D binary
    grating: maps the four regions around the ridge/groove corner to the vertex
    ordering ``(sup, ridge, sub, groove)`` so the diagonal pairs are
    ``(sup, sub)`` and ``(ridge, groove)`` (Li & Granet Fig. 1)."""
    return grating_convergence_class(
        (eps_superstrate, eps_ridge, eps_substrate, eps_groove))


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
covariant Li-1999 oblique-coordinate path, factorization='covariant', reaches the
spectral ~1e-7 floor by making the wall a coordinate surface.)
'''


# Register the PMM module caches with the library cache registry (so the global
# "clear all caches" path empties them too).  Canonical v4.16.0 enrollment pattern
# (mirrors rcwa.py and propagators/propagation.py).
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer(
        "pmm_lagrange_dref",
        lambda: getattr(_sys.modules[__name__], "_clear_pmm_caches")(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass
