"""Body-of-Revolution (axisymmetric) PMM -- Phase 0 / Milestone 1 prototype.

Standalone, validated radial spectral-element eigensolver for the cylindrical
metric + r=0 axis singularity -- the foundation of the BOR-PMM solver
(docs/pmm_bor_axisymmetric_roadmap.md).  This module deliberately stays OUTSIDE
the package (the roadmap's Phase-0 de-risking gate) and isolates the two
genuinely-new, error-prone pieces with EXACT analytic oracles, before any
coupled-interface / open-boundary / cascade work.

Per azimuthal order m, the radial modes psi(r) of a homogeneous cylinder of
radius R solve the Bessel eigenproblem

    (1/r) d/dr(r dpsi/dr) - (m^2/r^2) psi + gamma^2 psi = 0,   r in [0, R],

with regularity at r=0.  Boundary at r=R selects the polarization:
  * Dirichlet psi(R)=0  -> TM-like (E_z) -> gamma R = j_{m,n}   (Bessel zeros)
  * Neumann  psi'(R)=0  -> TE-like (H_z) -> gamma R = j'_{m,n}  (deriv. zeros)

Both reproduce the analytic spectrum to ~1e-13 (spectral convergence), and the
eigenfunctions match J_m(gamma r) to ~1e-13.

Weak form (test v, measure r dr; integrate the stiffness by parts):
    gamma^2 (r psi, v) = (r psi', v') + m^2 (psi/r, v)
=>  A psi = gamma^2 M psi,
    A_ij = INT r phi_i' phi_j' dr + m^2 INT (1/r) phi_i phi_j dr,
    M_ij = INT r phi_i phi_j dr.

AXIS TREATMENT (the crux).  For m != 0 the (1/r) and (m^2/r^2) terms are
singular at r=0.  Two ingredients make it clean:
  (1) GAUSS-LEGENDRE assembly (interior quadrature points) -- 1/r is finite at
      every quadrature point, so the singular axis is never sampled;
  (2) drop the r=0 DOF (impose psi(0)=0), which both enforces the physical
      r^|m| regularity (psi -> 0 for m != 0) and discards the one divergent
      matrix entry (A_00 ~ INT dr/r) before the solve.  The retained basis
      functions vanish at the axis, so (1/r) phi_i phi_j ~ r is integrable.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def _gll_nodes_weights(degree):
    """Gauss-Lobatto-Legendre nodes + weights on [-1, 1] (degree+1 points):
    endpoints +-1 plus the interior roots of P'_degree; weights
    w_i = 2/(degree(degree+1) P_degree(x_i)^2)."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    Pn = np.polynomial.legendre.Legendre.basis(degree)
    if degree == 1:
        nodes = np.array([-1.0, 1.0])
    else:
        interior = Pn.deriv().roots()
        interior = np.sort(interior.real[np.abs(interior.imag) < 1e-12])
        nodes = np.concatenate([[-1.0], interior, [1.0]])
    w = 2.0 / (degree * (degree + 1) * Pn(nodes) ** 2)
    return nodes, w


def _lagrange_vals_derivs(ref_nodes, xq):
    """Lagrange basis values L_a(xq) and derivatives L_a'(xq) on ref_nodes."""
    p = len(ref_nodes)
    nq = len(xq)
    V = np.zeros((nq, p))
    D = np.zeros((nq, p))
    for a in range(p):
        for r, x in enumerate(xq):
            num = 1.0
            for k in range(p):
                if k != a:
                    num *= (x - ref_nodes[k]) / (ref_nodes[a] - ref_nodes[k])
            V[r, a] = num
            d = 0.0
            for j in range(p):
                if j == a:
                    continue
                prod = 1.0 / (ref_nodes[a] - ref_nodes[j])
                for k in range(p):
                    if k != a and k != j:
                        prod *= (x - ref_nodes[k]) / (ref_nodes[a] - ref_nodes[k])
                d += prod
            D[r, a] = d
    return V, D


def _graded_boundaries(a, b, n_el):
    """Uniform element boundaries on [a, b]."""
    return np.linspace(a, b, n_el + 1)


def radial_spectrum(m, R, degree, n_el, *, bc="dirichlet", n_low=6,
                    return_modes=False, nq_extra=10):
    """Smallest ``n_low`` radial eigenvalues gamma^2 of a homogeneous cylinder
    of radius ``R``, azimuthal order ``m``.

    ``bc='dirichlet'`` (TM, psi(R)=0 -> Bessel zeros j_{m,n}) or ``'neumann'``
    (TE, natural psi'(R)=0 -> derivative zeros j'_{m,n}).  Returns the sorted
    eigenvalues, or ``(eigvals, nodal_eigvecs, r_nodes)`` if ``return_modes``.
    """
    ref_nodes, _w = _gll_nodes_weights(degree)
    p = degree + 1
    bnds = _graded_boundaries(0.0, R, n_el)
    n_el_a = len(bnds) - 1
    l2g = np.zeros((n_el_a, p), dtype=int)
    gid = 0
    for e in range(n_el_a):
        for a in range(p):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, p - 1]
            else:
                l2g[e, a] = gid
                gid += 1
    n_glob = gid
    nq = degree + nq_extra
    xq, wq = leggauss(nq)                    # Gauss-Legendre (interior pts)
    V, D = _lagrange_vals_derivs(ref_nodes, xq)
    A = np.zeros((n_glob, n_glob))
    M = np.zeros((n_glob, n_glob))
    r_glob = np.zeros(n_glob)
    for e in range(n_el_a):
        xl, xr = bnds[e], bnds[e + 1]
        J = 0.5 * (xr - xl)
        rq = 0.5 * (xr + xl) + J * xq
        wp = wq * J
        Dp = D / J
        idx = l2g[e]
        r_glob[idx] = 0.5 * (xr + xl) + J * ref_nodes
        for a in range(p):
            for b in range(p):
                A[idx[a], idx[b]] += (np.sum(wp * rq * Dp[:, a] * Dp[:, b])
                                      + m * m * np.sum(wp * (1.0 / rq)
                                                       * V[:, a] * V[:, b]))
                M[idx[a], idx[b]] += np.sum(wp * rq * V[:, a] * V[:, b])
    drop = set()
    if bc == "dirichlet":
        drop.add(n_glob - 1)
    elif bc != "neumann":
        raise ValueError("bc must be 'dirichlet' or 'neumann'")
    if m != 0:
        drop.add(0)                          # axis DOF: psi(0)=0 for m != 0
    keep = np.array([i for i in range(n_glob) if i not in drop])
    Ak = A[np.ix_(keep, keep)]
    Mk = M[np.ix_(keep, keep)]
    if return_modes:
        w, vec = np.linalg.eig(np.linalg.solve(Mk, Ak))
        order = np.argsort(w.real)
        w = w[order].real
        full = np.zeros((n_glob, vec.shape[1]))
        full[keep] = vec.real[:, order]
        return w[:n_low], full[:, :n_low], r_glob
    w = np.sort(np.linalg.eigvals(np.linalg.solve(Mk, Ak)).real)
    return w[:n_low]
