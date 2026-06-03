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

Scope (first release)
---------------------
1-D binary grating, normal OR oblique incidence (``angle`` adds the ``+i kx0``
Bloch shift of the pseudo-periodic envelope; the forward modes are then chosen
by the z-Poynting flux).  NumPy / SciPy (dense generalized eig); not
JAX-differentiable.  TM converges monotone-no-floor but only spectral-*ish* (the
matched TM partner ``Ex = q (1/eps) Hy`` is discontinuous at the wall and a C0
nodal value averages that jump) -- mesh grading toward the walls (multiple
graded elements per region) recovers the corner resolution and is the speed
lever for metal TM.

Anisotropic (Jones) path
------------------------
:func:`pmm_jones_1d` extends the solver to a binary grating whose ridge / groove
are full ``(3, 3)`` IN-PLANE permittivity tensors (the tunable-LC reflective
grating), returning the full complex ``2x2`` Jones reflection -- the
spectral-element counterpart of :func:`~lumenairy.elements.rcwa.rcwa_jones_1d`.
The modal field becomes a 2-vector ``[E_x; E_y]`` per node; the off-diagonal
``exy`` couples them.  The Li-1996 factorization is realized in the nodal basis:
the wall-normal inverse rule ``[[1/exx]]^{-1}`` becomes ``inv(hat(1/exx))`` (the
nodal multiply-by-``1/exx`` operator, inverted), and the ``Kx``-derivative terms
(``Ez``-elimination and ``Kx^2``) become spectral-element STIFFNESS operators
weighted by ``1/ezz`` / ``1`` -- so the inverse rule is AUTOMATIC and exact (the
``eps`` jump is on an element boundary).  The coupled second-order modal operator
mirrors the FMM tensor block ``M = -P@Q``.  Normal OR oblique incidence (the
``kx0`` Bloch shift + a z-Poynting-flux forward selector), binary, NumPy only
(multi-region / autodiff are follow-ons).
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import Legendre

__all__ = ["pmm_efficiency_1d", "pmm_efficiency_1d_segments",
           "pmm_jones_1d", "pmm_jones_1d_segments", "PMMStack"]

_C = np.complex128

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

def _gll_nodes_weights(degree: int):
    """GLL nodes (``degree + 1`` of them) and quadrature weights on ``[-1, 1]``:
    endpoints +/-1 plus the roots of ``P_degree'``; ``w_i = 2 / (degree
    (degree+1) P_degree(x_i)^2)``."""
    if degree == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    interior = np.sort(Legendre.basis(degree).deriv().roots().real)
    nodes = np.concatenate([[-1.0], interior, [1.0]])
    PD = Legendre.basis(degree)
    w = 2.0 / (degree * (degree + 1) * (PD(nodes) ** 2))
    return nodes, w


def _lagrange_derivative_matrix(nodes):
    """Differentiation matrix ``D[i, j] = l_j'(x_i)`` for the nodal Lagrange
    basis (barycentric form)."""
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

    kz_sup, kz_sub = _kz_forward(eps_sup, kx), _kz_forward(eps_sub, kx)
    kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
    if polarization == "te":
        R = np.real(kz_sup / kz_inc) * np.abs(r_ord) ** 2
        T = np.real(kz_sub / kz_inc) * np.abs(t_ord) ** 2
    else:
        flux_inc = np.real(kz_inc / eps_sup)
        R = np.real(kz_sup / eps_sup) * np.abs(r_ord) ** 2 / flux_inc
        T = np.real(kz_sub / eps_sub) * np.abs(t_ord) ** 2 / flux_inc
    R = np.where(np.real(kz_sup) > 1e-12, np.real(R), 0.0)
    T = np.where(np.real(kz_sub) > 1e-12, np.real(T), 0.0)
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
    safe_r = np.where(np.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = np.where(np.abs(kz_sub) < 1e-12, 1.0, kz_sub)

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
        # longitudinal Ez from div D = 0 in the isotropic half-space (rz =
        # -kx rx / kz); it carries z-flux for the wall-normal (Ex / TM-like)
        # component -- the term that makes the TM channel conserve energy.
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        # per-COLUMN incident flux: the col-0 wave (incident Ex=1, p-pol) ALSO
        # carries Ez_inc = -kx0 Ex/kz_inc, so its z-flux is kz_inc(1+(kx0/kz_inc)
        # ^2), NOT kz_inc; col-1 (Ey, s-pol) has Ez_inc=0.  At kx0=0 both reduce
        # to kz_inc -> bit-identical to the normal-incidence solve.
        flux_inc = kz_inc * (1.0 + (kx0n / kz_inc) ** 2) if col == 0 else kz_inc
        Re = np.real(kz_sup) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                + np.abs(rz) ** 2) / flux_inc
        Te = np.real(kz_sub) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                + np.abs(tz) ** 2) / flux_inc
        R_eff[col] = np.where(np.real(kz_sup) > 1e-12, np.real(Re), 0.0)
        T_eff[col] = np.where(np.real(kz_sub) > 1e-12, np.real(Te), 0.0)
        jones[0, col] = rx[m0]                  # PUBLIC convention -> no conjugation
        jones[1, col] = ry[m0]
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

    The ridge and groove are full ``(3, 3)`` IN-PLANE permittivity tensors (the
    tunable-LC reflective grating); the off-diagonal ``exy`` couples ``E_x`` and
    ``E_y`` in the spectral-element modal eigenproblem, so the response is a full
    Jones matrix (the phase relationship the scalar :func:`pmm_efficiency_1d`
    cannot carry).  Converges SPECTRALLY in the polynomial ``degree`` with no
    accuracy floor -- the PMM win on metals where the FMM needs many orders and
    the ASR stretch plateaus.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        Permittivity tensors of the ridge / groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an isotropic
        region; build LC tensors with
        :func:`~lumenairy.elements.rcwa.uniaxial_tensor` (``theta = pi/2`` keeps
        the director in-plane).  Must be IN-PLANE (no ``eps_xz / eps_yz /
        eps_zx / eps_zy``); an out-of-plane tensor raises ``ValueError`` (use
        :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` for the full-3x3 case).
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
    NumPy / SciPy (dense generalized eig); not JAX-differentiable.  Normal or
    oblique incidence, binary grating, in-plane tensor only (multi-region /
    autodiff are follow-ons).
    """
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d: degree must be >= 2.")
    if not (0.0 <= float(duty_cycle) <= 1.0):
        raise ValueError(
            f"pmm_jones_1d: duty_cycle must be in [0, 1], got {duty_cycle}.")
    er = np.asarray(eps_ridge, dtype=_C)
    eg = np.asarray(eps_groove, dtype=_C)
    if er.shape[-2:] != (3, 3) or eg.shape[-2:] != (3, 3):
        raise ValueError(
            "pmm_jones_1d: eps_ridge / eps_groove must be (3, 3) permittivity "
            "tensors (use scalar * np.eye(3) for an isotropic region).")
    # in-plane only: reject out-of-plane coupling (would be silently dropped)
    scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    off = max(float(np.max(np.abs(er[[0, 1, 2, 2], [2, 2, 0, 1]]))),
              float(np.max(np.abs(eg[[0, 1, 2, 2], [2, 2, 0, 1]]))))
    if off > 1e-9 * scale:
        raise ValueError(
            "pmm_jones_1d: the anisotropic PMM is the z-decoupled IN-PLANE "
            "tensor subset (exx, exy, eyx, eyy, ezz); the supplied tensor has "
            "out-of-plane coupling (eps_xz / eps_yz / eps_zx / eps_zy != 0). "
            "Use rcwa_jones_1d for the full-3x3 (out-of-plane) case.")

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
        UPWARD degree window and returns the minimum-power, resonance-free
        result -- build-reproducible and never below the requested degree's
        accuracy.  Set ``False`` to solve at exactly ``degree`` (e.g. for
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
    NumPy / SciPy (dense generalized eig); not JAX-differentiable.  TM converges
    monotone-with-no-floor but only spectral-*ish* (the discontinuous TM partner
    is C0-averaged at the wall); ``elements_per_region>1, grade=True`` recovers
    the rate for metals.
    """
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d: polarization must be 'te' or 'tm', got "
            f"{polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d: degree must be >= 2.")
    if not (0.0 <= float(duty_cycle) <= 1.0):
        raise ValueError(
            f"pmm_efficiency_1d: duty_cycle must be in [0, 1], got "
            f"{duty_cycle}.")

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
    # in-plane only: reject out-of-plane coupling (would be silently dropped)
    scale = max([float(np.max(np.abs(M))) for M in tensors] + [1.0])
    off = max(float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]])))
              for M in tensors)
    if off > 1e-9 * scale:
        raise ValueError(
            "pmm_jones_1d_segments: the anisotropic PMM is the z-decoupled "
            "IN-PLANE tensor subset (exx, exy, eyx, eyy, ezz); a segment has "
            "out-of-plane coupling (eps_xz / eps_yz / eps_zx / eps_zy != 0). "
            "Use rcwa_jones_1d_segments for the full-3x3 case.")
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
    """``(3,3)`` (or scalar) permittivity -> the ``dict(exx, exy, eyx, eyy, ezz)``
    the tensor spectral-element assembler consumes."""
    M = np.asarray(M, dtype=_C)
    if M.ndim == 0:
        M = M * np.eye(3, dtype=_C)
    return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1], ezz=M[2, 2])


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

    Notes
    -----
    Anisotropic / Jones throughout (scalar layers are promoted to isotropic
    tensors), IN-PLANE tensors only (use :class:`RCWAStack` for out-of-plane),
    normal or oblique incidence, NumPy (not JAX).  The modal forward set uses the
    z-Poynting-flux selector (as the multi-region single-layer solver), so the
    many-element shared grid stays resonance-free.
    """

    def __init__(self, period, *, n_substrate=1.0, n_superstrate=1.0,
                 degree=16, elements_per_region=1, grade=True,
                 far_field_orders=21):
        if int(degree) < 2:
            raise ValueError("PMMStack: degree must be >= 2.")
        self.period = float(period)
        self.n_sub = _C(n_substrate)
        self.n_sup = _C(n_superstrate)
        self.degree = int(degree)
        self.n_el = int(elements_per_region)
        self.grade = bool(grade)
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
        scale = max(float(np.max(np.abs(M))), 1.0)
        off = float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]])))
        if off > 1e-9 * scale:
            raise ValueError(
                "PMMStack: the anisotropic PMM is the z-decoupled IN-PLANE tensor "
                "subset (exx, exy, eyx, eyy, ezz); a layer has out-of-plane "
                "coupling (eps_xz / eps_yz / eps_zx / eps_zy != 0).  Use RCWAStack "
                "for the full-3x3 case.")
        return M

    def add_layer(self, thickness, *, segments=None, eps=None):
        """Append a layer.  Give exactly one of ``eps`` (uniform: scalar or
        ``(3,3)`` tensor) or ``segments`` (a list of ``(width_fraction, eps)`` --
        each ``eps`` scalar or ``(3,3)``; widths sum to 1).  Returns ``self``."""
        if (segments is None) == (eps is None):
            raise ValueError(
                "PMMStack.add_layer: give exactly one of `segments` or `eps`.")
        if eps is not None:
            segs = [(1.0, self._as_tensor(eps))]
        else:
            if len(segments) < 1:
                raise ValueError("PMMStack.add_layer: empty segments.")
            segs = [(float(w), self._as_tensor(e)) for w, e in segments]
        self._layers.append((float(thickness), segs))
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
        lmodes = [_sem_modes_tensor(m, k0, kx0, True) for m in layer_mats]

        # Redheffer recursion: sup -> [interface, propagation]*layers -> sub
        S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
        for i, (Wl, Vl, lam_l, _q) in enumerate(lmodes):
            S = _redheffer_star(S, _propagation_smatrix(
                lam_l, k0 * self._layers[i][0]))
            nW, nV = ((Wsub, Vsub) if i == len(lmodes) - 1
                      else (lmodes[i + 1][0], lmodes[i + 1][1]))
            S = _redheffer_star(S, _interface_smatrix(Wl, Vl, nW, nV))
        S11, _S12, S21, _S22 = S

        # far-field projection (mirrors _pmm_jones_solve_core)
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
        m_prop = _n_propagating_orders(self.period, wl, n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
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
        safe_r = np.where(np.abs(kz_sup) < 1e-12, 1.0, kz_sup)
        safe_t = np.where(np.abs(kz_sub) < 1e-12, 1.0, kz_sub)
        m0 = np.where(orders == 0)[0][0]
        jones = np.zeros((2, 2), dtype=_C)
        R_eff = np.zeros((2, N))
        T_eff = np.zeros((2, N))
        for col in range(2):
            rhs = np.zeros(2 * N, dtype=_C)
            rhs[(col * N) + m0] = 1.0
            cinc, *_ = np.linalg.lstsq(Hsup, rhs, rcond=None)
            r_ord = Hsup @ (S11 @ cinc)
            t_ord = Hsub @ (S21 @ cinc)
            rx, ry = r_ord[:N], r_ord[N:]
            tx, ty = t_ord[:N], t_ord[N:]
            rz = -(kx * rx) / safe_r
            tz = -(kx * tx) / safe_t
            flux_inc = (kz_inc * (1.0 + (kx0n / kz_inc) ** 2) if col == 0
                        else kz_inc)
            Re = np.real(kz_sup) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                    + np.abs(rz) ** 2) / flux_inc
            Te = np.real(kz_sub) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                    + np.abs(tz) ** 2) / flux_inc
            R_eff[col] = np.where(np.real(kz_sup) > 1e-12, np.real(Re), 0.0)
            T_eff[col] = np.where(np.real(kz_sub) > 1e-12, np.real(Te), 0.0)
            jones[0, col] = rx[m0]
            jones[1, col] = ry[m0]
        return orders, R_eff, T_eff, jones
