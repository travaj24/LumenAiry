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
1-D binary grating, **normal incidence only** (``angle == 0``; oblique needs the
``+i kx0`` Bloch shift in the stiffness and is not yet implemented -- raises
``NotImplementedError``).  NumPy / SciPy (dense generalized eig); not
JAX-differentiable.  TM converges monotone-no-floor but only spectral-*ish* (the
matched TM partner ``Ex = q (1/eps) Hy`` is discontinuous at the wall and a C0
nodal value averages that jump) -- mesh grading toward the walls (multiple
graded elements per region) recovers the corner resolution and is the speed
lever for metal TM.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import Legendre

__all__ = ["pmm_efficiency_1d"]

_C = np.complex128


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
    for e in range(n_el):
        xl, xr, eps = elem_bnds[e]
        J = 0.5 * (xr - xl)                         # dx/dxi
        inv = 1.0 / eps
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)                         # GLL mass (diagonal)
        Kloc = (Dphys.T * wel) @ Dphys              # stiffness
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        S0[ix] += Mloc
        Peps[ix] += eps * Mloc
        Pinv[ix] += inv * Mloc
        L[ix] += Kloc
        Linv[ix] += inv * Kloc
    return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, n_glob=n_glob,
                l2g=l2g, elem_bnds=elem_bnds, degree=degree, ref_nodes=ref_nodes)


def _sem_modes(mats, k0, polarization):
    """Periodic generalized eigenproblem on the nodal basis.

    Returns ``(Acoef, lam, q, invop)``: ``Acoef[:, n]`` = nodal values of mode
    ``n``'s field profile (``E_y`` TE / ``H_y`` TM); ``q = gamma/k0`` with
    ``Im(q) >= 0``; ``lam = -i q`` (forward-decaying propagator); ``invop`` =
    nodal multiply-by-``1/eps`` operator (TM only).
    """
    k02 = k0 * k0
    if polarization == "te":
        A, B = mats["Peps"] - mats["L"] / k02, mats["S0"]
        invop = None
    else:
        A, B = mats["S0"] - mats["Linv"] / k02, mats["Pinv"]
        invop = np.linalg.solve(mats["S0"], mats["Pinv"])
    q2, Acoef = sla.eig(A, B)
    q = np.sqrt(q2)
    q = np.where(q.imag < 0.0, -q, q)               # Im(q) >= 0 forward decay
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
               far_field_orders):
    eps_ridge, eps_groove = n_ridge ** 2, n_groove ** 2
    eps_sup, eps_sub = n_sup ** 2, n_sub ** 2
    k0 = 2.0 * np.pi / wl
    d_wall = duty * period

    mats = _build_sem(period, d_wall, eps_ridge, eps_groove, degree,
                      n_ridge_el, n_groove_el, grade)
    mats_sup = _build_sem(period, d_wall, eps_sup, eps_sup, degree,
                          n_ridge_el, n_groove_el, grade)
    mats_sub = _build_sem(period, d_wall, eps_sub, eps_sub, degree,
                          n_ridge_el, n_groove_el, grade)
    n_glob = mats["n_glob"]

    # Rayleigh order set for the (forward-only) far-field projection: cover the
    # propagating orders with an evanescent buffer, kept WELL BELOW n_glob (a
    # projection order count approaching n_glob aliases the nodal->Fourier map).
    n_max = max(np.real(n_sup), np.real(n_sub), np.real(n_ridge),
                np.real(n_groove))
    m_prop = _n_propagating_orders(period, wl, n_max)
    n_proj = max(int(far_field_orders), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"pmm_efficiency_1d: degree={degree} too low to resolve the "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
            f"degree or elements_per_region.")
    orders = np.arange(-half, half + 1)
    G = 2.0 * np.pi / period
    kx = (orders * G) / k0
    Tp = _sem_fourier_projection(orders, period, mats)

    Acoef, lam_l, q_l, invop = _sem_modes(mats, k0, polarization)
    Wl = Acoef
    Vl = (Acoef if polarization == "te" else invop @ Acoef) @ np.diag(q_l)

    Wsup, _ls, q_sup, invsup = _sem_modes(mats_sup, k0, polarization)
    Wsub, _lb, q_sub, invsub = _sem_modes(mats_sub, k0, polarization)
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
    kz_inc = float(np.real(_kz_forward(eps_sup, np.array([0.0]))[0]))
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
        Incidence angle (radians).  **Only ``0`` (normal) is implemented**;
        a non-zero angle raises ``NotImplementedError`` (oblique needs the
        ``+i kx0`` Bloch shift in the stiffness).
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
        Retry nearby ``degree`` when an isolated near-singular layer<->region
        resonance pushes ``sum(R)+sum(T) > 1`` (a measure-zero erratic event,
        the analogue of the FMM ``stabilize`` flag).  Default ``True``.

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
    if abs(float(angle)) > 1e-12:
        raise NotImplementedError(
            "pmm_efficiency_1d: only normal incidence (angle=0) is implemented "
            "(oblique needs the +i*kx0 Bloch shift in the element stiffness).")
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
              far_field_orders=int(far_field_orders))

    bumps = (0, 1, -1, 2, 3, -2, 5, 7) if stabilize else (0,)
    last = None
    for bump in bumps:
        d = int(degree) + bump
        if d < 2:
            continue
        orders, R, T, _ = _pmm_solve(*args, degree=d, **kw)
        last = (orders, R, T)
        if float(R.sum() + T.sum()) <= 1.0 + 1e-6:    # passive: <= 1 (+eps)
            return orders, R, T
    return last                                        # best effort if none pass
