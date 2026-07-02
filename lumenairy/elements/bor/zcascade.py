"""BOR-PMM Milestone 4: z-cascade S-matrix for axisymmetric stacks.

A stack is a sequence of z-layers, each with eps = eps(r) on a SHARED radial
grid.  Each layer's radial vector modes (from the M2 solver) form a modal basis;
tangential-field continuity at the z-interfaces + modal propagation through each
layer, cascaded by the Redheffer star, gives the stack S-matrix.

Key facts established (each validated independently before this module):
- Modal tangential fields: ``W = [E_r; E_phi]`` (tangential E), ``V = [h_r; h_phi]``
  (tangential H), with
    h_r   = (1/k0)[(m/r) E_z - q E_phi]
    h_phi = (1/k0)[q E_r + i dE_z/dr]      (E_z = q * Phi from the M2 elimination)
  -> satisfies curl h = -i k0 eps E to ~1e-12 (with the inverse-rule normal flux).
- BACKWARD mode = ``[W; -V]`` EXACTLY (q -> -q flips E_z, hence h_t, leaving E_t):
  so lumenairy's ``_interface_smatrix`` (which hardwires that symmetry) is reusable
  VERBATIM.  The interface match is POINTWISE (Wb^{-1} Wa) on the shared grid -- the
  r*dr measure does NOT enter the interface algebra, only the flux/energy accounting.
- Forward/backward split: propagating modes by the sign of the r*dr z-flux
  P_z = Re sum (E_r h_phi* - E_phi h_r*) r dr; evanescent by Im(q) > 0.

This is the FD prototype (the M2/M3 discretization); a SEM re-discretization is the
production accuracy upgrade.  Validated against analytic anchors in
``test_zcascade.py`` (same-medium identity, per-mode Fresnel, slab Fabry-Perot,
r*dr energy conservation).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from .coupled_radial_eigensolver import (
    _assemble_staggered,
    _fast_geig,
    _fd_grid,
    _normal_eps,
    _pec_wall_ops,
    _pml_stretch,
)


# --------------------------------------------------------------------------- #
#  Layer modes -> (W, V, q) with forward orientation                           #
# --------------------------------------------------------------------------- #
def _layer_modes_staggered(m, Rbig, N, eps_profile, k0):
    """Forward modal basis on the Yee div-conforming grid (spurious-mode cure).
    ``W = [E_r(faces); E_phi(nodes)]``; the DUAL Yee H-placement puts
    ``h_r`` on NODES (with E_phi) and ``h_phi`` on FACES (with E_r) so the
    z-Poynting flux ``E_r h_phi* (faces) - E_phi h_r* (nodes)`` pairs WITHOUT
    interpolation -> machine-precision energy conservation in the cascade.

    Grid bookkeeping (audit P3-14): rows ``[:N]`` of ``W``/``V`` live on the
    FACE grid ``r_face = (i+1)h`` and rows ``[N:]`` on the NODE grid
    ``r = (i+1/2)h``.  Both grids and their per-half ``r*dr`` quadrature
    weights are returned (``r``/``wq_node`` for the node half, ``r_face``/
    ``wq_face`` for the face half); the legacy ``wq`` key equals ``wq_node``
    and must NOT be applied to the face rows (an O(h) half-cell-shift
    error)."""
    op = _assemble_staggered(m, Rbig, N, eps_profile, k0)
    Lei, A_f2n, Dn2f = op["Lei"], op["A_f2n"], op["Dn2f"]
    mrn = op["mrn"]
    r_n, r_f, h = op["r_n"], op["r_f"], op["h"]
    q2, Vm = _fast_geig(op["K"], op["B"])
    q = np.sqrt(q2)

    def hfields(Er, Ephi, qj):
        Ez = qj * (Lei @ (1j * A_f2n @ Er - mrn * Ephi))   # nodes
        hr = (1.0 / k0) * (mrn * Ez - qj * Ephi)           # NODES (with E_phi)
        hphi = (1.0 / k0) * (qj * Er + 1j * (Dn2f @ Ez))   # FACES (with E_r)
        return hr, hphi

    def flux(Er, Ephi, hr, hphi):
        return np.real(np.sum(Er * np.conj(hphi) * r_f * h)
                       - np.sum(Ephi * np.conj(hr) * r_n * h))

    nm = len(q)
    W = np.zeros((2 * N, nm), dtype=complex)
    V = np.zeros((2 * N, nm), dtype=complex)
    qf = np.zeros(nm, dtype=complex)
    for j in range(nm):
        Er, Ephi = Vm[:N, j], Vm[N:, j]
        qj = q[j]
        hr, hphi = hfields(Er, Ephi, qj)
        P = flux(Er, Ephi, hr, hphi)
        if abs(qj.imag) < 1e-9 * max(abs(qj.real), 1e-300):
            qj = qj if P >= 0 else -qj
        else:
            qj = qj if qj.imag > 0 else -qj
        hr, hphi = hfields(Er, Ephi, qj)
        P = flux(Er, Ephi, hr, hphi)
        # propagating-vs-evanescent split by flux RELATIVE to the mode's own
        # r*dr field norm (both scale as field^2*length^2 -> unit-invariant;
        # an absolute threshold silently mis-normalized meter-scale inputs)
        fnrm = np.sum(np.abs(Er) ** 2 * r_f * h) + np.sum(np.abs(Ephi) ** 2 * r_n * h)
        s = (1.0 / np.sqrt(abs(P)) if abs(P) > 1e-10 * fnrm
             else 1.0 / np.sqrt(np.sum(np.abs(Er) ** 2 + np.abs(Ephi) ** 2) + 1e-300))
        W[:N, j] = Er * s
        W[N:, j] = Ephi * s
        V[:N, j] = hr * s
        V[N:, j] = hphi * s
        qf[j] = qj
    wq_node = (r_n * h).astype(complex)
    wq_face = (r_f * h).astype(complex)
    return dict(W=W, V=V, q=qf, r=r_n, wq=wq_node, N=N,
                r_face=r_f, wq_node=wq_node, wq_face=wq_face)


def layer_modes(m, Rbig, N, eps_profile, k0, *, R_pml=None, sigma_max=5.0,
                pml_p=2, wall="natural", staggered=False):
    """Return the forward modal basis of a layer: ``W`` (2N x 2N tangential E),
    ``V`` (2N x 2N tangential H), ``q`` (2N axial wavenumbers, forward-oriented),
    plus the shared grid ``r`` and ``r*dr`` weights ``wq``.

    ``wall='pec'`` (M5a) imposes a Dirichlet (closed) outer wall -> clean real-q
    box modes (needed for physical multi-mode half-spaces); ``'natural'`` is the
    leaky one-sided wall (only the bound modes are clean).  ``R_pml`` overrides
    to an open PML boundary.  ``staggered=True`` uses the Yee div-conforming
    discretization (the spurious-mode CURE -> machine-precision cascade energy);
    the closed Dirichlet wall is built into its node->face stencil.

    .. note:: (audit P3-14) With ``staggered=True`` the basis is TWO-grid:
       rows ``[:N]`` of ``W``/``V`` (``E_r``, ``h_phi``) live on the FACE grid
       and rows ``[N:]`` (``E_phi``, ``h_r``) on the NODE grid.  ``r``/``wq``
       describe the NODE grid only; the extra keys ``r_face``/``wq_face`` (and
       the alias ``wq_node = wq``) carry the face grid + per-half quadrature.
       Flux/far-field quadrature must weight the face rows by ``wq_face`` and
       the node rows by ``wq_node`` (a single shared ``wq`` is only correct
       for ``staggered=False``, where all rows share one cell-centered grid).
    """
    if staggered:
        return _layer_modes_staggered(m, Rbig, N, eps_profile, k0)
    r, D, h = _fd_grid(Rbig, N)
    eps = np.asarray(eps_profile(r), dtype=complex)
    eps_n = _normal_eps(eps)
    Lap = None
    if R_pml is not None:
        sinv, rg = _pml_stretch(r, h, R_pml, Rbig, sigma_max, pml_p)
        D = np.diag(sinv) @ D
    else:
        rg = r
        if wall == "pec":
            D, Lap = _pec_wall_ops(D, h, N)
    Im = np.eye(N)
    ir = np.diag(1.0 / rg)
    mr = m * ir
    m2r2 = (m ** 2) * np.diag(1.0 / rg ** 2)
    A = D + ir
    Lm = (D @ D if Lap is None else Lap) + ir @ D - m2r2
    dA = D @ A
    Lei = np.linalg.inv(Lm + k0 ** 2 * np.diag(eps))
    Phi_r = Lei @ (1j * A)
    Phi_p = Lei @ (-mr)
    B = np.block([[Im + 1j * D @ Phi_r, 1j * D @ Phi_p],
                  [-mr @ Phi_r,         Im - mr @ Phi_p]])
    K = np.block([[k0 ** 2 * np.diag(eps_n) - m2r2, -1j * mr @ A],
                  [-1j * D @ mr,        k0 ** 2 * np.diag(eps) + dA]])
    q2, Vm = eig(K, B)
    q = np.sqrt(q2)
    wq = (r * h).astype(complex)                      # r*dr quadrature weights
    nm = len(q)
    W = np.zeros((2 * N, nm), dtype=complex)
    V = np.zeros((2 * N, nm), dtype=complex)
    qf = np.zeros(nm, dtype=complex)
    for j in range(nm):
        Er = Vm[:N, j]
        Ephi = Vm[N:, j]
        qj = _orient_forward(q[j], Er, Ephi, D, mr, Lei, A, k0, wq)
        Ez = qj * (Lei @ (1j * A @ Er - mr @ Ephi))
        hr = (1.0 / k0) * (mr @ Ez - qj * Ephi)
        hphi = (1.0 / k0) * (qj * Er + 1j * (D @ Ez))
        W[:N, j] = Er
        W[N:, j] = Ephi
        V[:N, j] = hr
        V[N:, j] = hphi
        qf[j] = qj
    return dict(W=W, V=V, q=qf, r=r, wq=wq, N=N)


def _orient_forward(q, Er, Ephi, D, mr, Lei, A, k0, wq):
    """Orient q so the mode is forward (+z): propagating by r*dr flux sign,
    evanescent by Im(q) > 0."""
    def flux(qq):
        Ez = qq * (Lei @ (1j * A @ Er - mr @ Ephi))
        hr = (1.0 / k0) * (mr @ Ez - qq * Ephi)
        hphi = (1.0 / k0) * (qq * Er + 1j * (D @ Ez))
        return np.real(np.sum((Er * np.conj(hphi) - Ephi * np.conj(hr)) * wq))
    if abs(q.imag) < 1e-9 * max(abs(q.real), 1e-300):     # propagating
        return q if flux(q) >= 0 else -q
    return q if q.imag > 0 else -q                         # evanescent: decay +z


# --------------------------------------------------------------------------- #
#  S-matrix primitives (mirror lumenairy rcwa/_core.py; backward == [W; -V])    #
# --------------------------------------------------------------------------- #
def interface_smatrix(Wa, Va, Wb, Vb):
    """Interface a -> b.  Tangential E,H continuity, backward modes = [W; -V]."""
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    apb = a + b
    amb = a - b
    iapb = np.linalg.inv(apb)
    return (-iapb @ amb, 2.0 * iapb,
            0.5 * (apb - amb @ iapb @ amb), amb @ iapb)


def propagation_smatrix(q, L):
    """Pure propagation through thickness L: forward/backward modes acquire
    X = exp(i q L) (phase for propagating, decay for evanescent), no reflection.
    (Forward q is oriented so Im(q) >= 0 -> exp(i q L) decays.)"""
    X = np.diag(np.exp(1j * q * L))
    Z = np.zeros_like(X)
    return (Z, X, X, Z)


def redheffer_star(SA, SB):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=complex)
    D = np.linalg.inv(I - B11 @ A22)
    F = np.linalg.inv(I - A22 @ B11)
    return (A11 + A12 @ D @ B11 @ A21,
            A12 @ D @ B12,
            B21 @ F @ A21,
            B22 + B21 @ F @ A22 @ B12)


def cascade(layers, k0):
    """Cascade a list of layers.  Each layer is a dict with its modal basis
    (from ``layer_modes``) and a ``thickness`` (None / inf for semi-infinite
    end half-spaces).  Returns the global S-matrix (S11,S12,S21,S22).

    Build: interface(L0,L1) * prop(L1) * interface(L1,L2) * prop(L2) * ...
    The two end layers are semi-infinite (no propagation S-matrix)."""
    S = interface_smatrix(layers[0]["W"], layers[0]["V"],
                          layers[1]["W"], layers[1]["V"])
    for i in range(1, len(layers) - 1):
        Li = layers[i]
        S = redheffer_star(S, propagation_smatrix(Li["q"], Li["thickness"]))
        S = redheffer_star(S, interface_smatrix(Li["W"], Li["V"],
                                                layers[i + 1]["W"],
                                                layers[i + 1]["V"]))
    return S
