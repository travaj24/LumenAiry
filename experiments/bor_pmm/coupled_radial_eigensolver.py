"""BOR-PMM Milestone 2: coupled (E_r, E_phi) radial vector eigensolver.

Solves the cylindrical waveguide eigenproblem for eps = eps(r) (one azimuthal
order m, fields ~ exp(i m phi + i q z)) and returns the propagation constants q
and modal fields.  The three pieces that make it correct (each validated against
the open-cladding fiber oracle, ``fiber_oracle.py``):

1. **q^2 formulation with E_z elimination.**  Maxwell -> a generalized
   eigenproblem ``K Psi = q^2 B Psi`` in ``Psi = (E_r, E_phi)``, with the
   longitudinal ``E_z`` eliminated through ``Phi = (L_m + k0^2 eps)^{-1}[...]``
   (the cylindrical analog of the Cartesian PMM ``G = I - Kx(1/ezz)Kx``
   elimination in ``lumenairy/elements/pmm/_core.py:_sem_modes_tensor``).  This
   linear-in-``q^2`` form makes the solve a single dense ``eig``.

2. **Wall-normal inverse rule.**  At a ring interface ``D_r = eps E_r`` is
   continuous while ``E_r`` jumps; the normal-component eps uses the harmonic
   mean ([[1/eps]]^{-1}) rather than the pointwise value.  This removes the
   interface mode-doubling and sharpens the guided q to the oracle.  (Mirrors
   ``Cxx = [[1/exx]]^{-1}`` in the Cartesian solver; tangential eps stays
   pointwise.)

3. **Divergence-free filter.**  Real-space vector discretizations emit spurious
   modes that violate ``div(eps E) = 0``.  Physical modes have tiny relative
   divergence (~1e-2); spurious ones are O(1-10).  ``relative_divergence`` below
   flags them cleanly (>150x separation on the fiber test).

The radial operator uses a cell-centered grid (nodes at ``(i+1/2)h`` -> never
samples ``r = 0``, so the axis 1/r is regular; cf. the M1 Gauss-quadrature axis
treatment).  Convergence is 2nd-order in N (FD); a spectral-element upgrade is
the accuracy follow-on, but the FD form already matches the oracle to ~1e-4..1e-5.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig


def _fd_grid(Rbig, N):
    """Cell-centered grid + 2nd-order differentiation matrix on (0, Rbig)."""
    h = Rbig / N
    r = (np.arange(N) + 0.5) * h
    D = np.zeros((N, N))
    for i in range(1, N - 1):
        D[i, i - 1] = -1.0 / (2 * h)
        D[i, i + 1] = 1.0 / (2 * h)
    D[0, 0] = -1.0 / h           # one-sided at the ends
    D[0, 1] = 1.0 / h
    D[N - 1, N - 2] = -1.0 / h
    D[N - 1, N - 1] = 1.0 / h
    return r, D, h


def _normal_eps(eps):
    """Wall-normal effective eps: harmonic mean across each eps jump
    ([[1/eps]]^{-1} inverse rule), pointwise elsewhere."""
    en = (1.0 / eps).copy()
    for i in range(1, len(eps)):
        if eps[i] != eps[i - 1]:
            hm = 2.0 / (1.0 / eps[i] + 1.0 / eps[i - 1])
            en[i] = en[i - 1] = 1.0 / hm
    return 1.0 / en


def radial_coupled_modes(m, Rbig, N, eps_profile, k0, *, inverse_rule=True):
    """All radial vector modes for azimuthal order ``m``.

    ``eps_profile`` : callable r-array -> eps-array (real or complex).
    Returns a list of dicts: ``q`` (propagation const), ``reldiv`` (relative
    |div(eps E)|, the spurious flag), ``Er``/``Ephi``/``Ez`` fields, ``r``.
    """
    r, D, _ = _fd_grid(Rbig, N)
    eps = np.asarray(eps_profile(r), dtype=complex)
    eps_n = _normal_eps(eps) if inverse_rule else eps
    I = np.eye(N)
    ir = np.diag(1.0 / r)
    mr = m * ir
    m2r2 = (m ** 2) * np.diag(1.0 / r ** 2)
    A = D + ir
    Lm = D @ D + ir @ D - m2r2
    dA = D @ A
    Lei = np.linalg.inv(Lm + k0 ** 2 * np.diag(eps))     # E_z elimination
    Phi_r = Lei @ (1j * A)
    Phi_p = Lei @ (-mr)
    B = np.block([[I + 1j * D @ Phi_r, 1j * D @ Phi_p],
                  [-mr @ Phi_r,        I - mr @ Phi_p]])
    K = np.block([[k0 ** 2 * np.diag(eps_n) - m2r2, -1j * mr @ A],
                  [-1j * D @ mr,        k0 ** 2 * np.diag(eps) + dA]])
    q2, Vm = eig(K, B)
    q = np.sqrt(q2)
    modes = []
    for j in range(len(q)):
        Er = Vm[:N, j]
        Ephi = Vm[N:, j]
        Ez = q[j] * (Lei @ (1j * A @ Er - mr @ Ephi))
        # div(eps E) using the CONSISTENT normal flux D_r = eps_n E_r (the same
        # inverse-rule eps the operator uses); pointwise eps on the tangential
        # components.  Using pointwise eps for D_r instead inflates the physical
        # modes' divergence ~100x and breaks the spurious/physical separation.
        Dr = eps_n * Er
        div = (1.0 / r) * (D @ (r * Dr)) + 1j * mr @ (eps * Ephi) + 1j * q[j] * (eps * Ez)
        En = np.sqrt(np.sum(np.abs(Er) ** 2 + np.abs(Ephi) ** 2 + np.abs(Ez) ** 2))
        reldiv = np.sqrt(np.sum(np.abs(div) ** 2)) / (k0 * max(En, 1e-300))
        modes.append(dict(q=q[j], reldiv=float(reldiv.real),
                          Er=Er, Ephi=Ephi, Ez=Ez, r=r))
    return modes


def guided_modes(m, a, Rbig, N, eps_core, eps_clad, k0, *,
                 reldiv_tol=1.0, tail_tol=0.05):
    """Bound guided modes (div-free AND decaying in the cladding), q descending."""
    def eps_profile(rr):
        return np.where(rr <= a, eps_core, eps_clad)
    out = []
    qlo, qhi = np.sqrt(eps_clad) * k0, np.sqrt(eps_core) * k0
    for md in radial_coupled_modes(m, Rbig, N, eps_profile, k0):
        q = md["q"]
        if not (qlo + 1e-2 < q.real < qhi - 1e-2 and abs(q.imag) < 1e-3):
            continue
        if md["reldiv"] > reldiv_tol:
            continue                                    # spurious
        amp = np.abs(md["Er"]) + np.abs(md["Ephi"])
        amp = amp / amp.max()
        if amp[md["r"] > 0.8 * Rbig].max() > tail_tol:
            continue                                    # radiation, not bound
        out.append(md)
    return sorted(out, key=lambda md: -md["q"].real)
