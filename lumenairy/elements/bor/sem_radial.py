"""Div-conforming radial SPECTRAL-ELEMENT basis for the BOR solver (the
"SEM follow-on" the axisymmetric roadmap gates the production solver on).

The cylindrical peer of the Cartesian PMM ``_sem_modes_tensor``: per azimuthal
order ``m`` and vacuum wavenumber ``k0``, solve the radial vector eigenproblem
for modes ``(E_r, E_phi)(r) ~ exp(i m phi + i q z)`` with ``E_z`` eliminated,
on ELEMENT MESHES ALIGNED WITH THE RING WALLS -- eps is constant per element,
so material interfaces are exact and convergence is spectral in the element
degree ``p`` (vs the 2nd-order floor of the Yee-staggered FD basis in
``coupled_radial_eigensolver``).

Compatible (de Rham) space pair -- the div-conforming cure realized at the
FUNCTION-SPACE level rather than by grid staggering:

* ``V0`` (``E_phi``, ``E_z``): C0 nodal Gauss-Lobatto-Legendre, degree ``p``.
  Tangential-to-ring-walls components are continuous -- enforced by shared
  element-boundary nodes, exactly as the physics demands.
* ``V1`` (``E_r``): DISCONTINUOUS per-element Gauss-Legendre nodal, degree
  ``p - 1`` (the 1-form/flux space; d/dr maps V0 onto V1).  ``E_r`` may jump
  at ring walls; ``D_r = eps_rr E_r`` continuity arrives weakly through the
  Galerkin form.  NO inverse rule is needed: the harmonic-mean device of the
  FD basis exists only because an FD cell can straddle a wall.

Weak form (measure ``r dr``, ``A = d/dr + 1/r``, derived from
``curl curl E = k0^2 eps E``; independently re-derived and cross-validated
against the staggered FD basis, the Bessel ladders and the analytic uniaxial
dispersion before landing)::

    K_rr = k0^2 eps_rr M_rr - m^2 N_rr        N_rr = <phi1, phi1 / r>
    K_rp = -i m C1,   K_pr = +i m C1^T        C1   = <phi1, A phi0>_dr
    K_pp = k0^2 eps_pp M_pp - S_A             S_A  = <A phi0, A phi0>_{r dr}
    Mz   = -S_z + k0^2 eps_zz M_00            (E_z block; S_z = M1 stiffness)
    E_z  = q * Mz^{-1} (-i D_zr E_r - m Q E_phi)
    B_rr = M_rr + D_zr^T Mz^{-1} D_zr         D_zr = <phi0', phi1>_{r dr}
    B_rp = -i m D_zr^T Mz^{-1} Q              Q    = <phi0, phi0>_dr
    B_pr = +i m Q^T  Mz^{-1} D_zr
    B_pp = M_pp + m^2 Q^T Mz^{-1} Q
    K Psi = q^2 B Psi

Anisotropy: the DIAGONAL CYLINDRICAL tensor diag(eps_rr, eps_phiphi, eps_zz)
per element -- each component enters exactly one role above (mirrors the
staggered-FD placement validated in tests/unit/test_bor_anisotropic.py).

Axis (r = 0) essential conditions (the M1 recipe, vector-generalized;
validated against the Bessel ladders per m): ``E_z(0) = 0`` for ``m != 0``;
``E_phi(0) = 0`` for ``m != 1`` (at ``|m| = 1`` the transverse field keeps a
finite axis value, cf. ``J_1(g r)/r -> g/2``); ``E_r`` has no axis DOF (the
V1 nodes are interior Gauss points) and the ``r dr`` measure handles the rest.
Assembly quadrature is Gauss-Legendre (interior points; ``1/r`` finite at
every point), over-integrated by ``nq_extra``.

PEC wall at ``Rbig``: ``E_phi(Rbig) = 0`` and ``E_z(Rbig) = 0`` essential;
``E_r`` free.

Cascade contract: ``sem_layer_modes`` returns the
``zcascade.layer_modes``-shaped dict (``W``, ``V``, ``q`` forward-oriented by
the SAME flux rule, unit-z-flux-normalized columns with the audit-P1-01
relative threshold) plus the block split ``n1`` -- the two component blocks
are NOT equal-sized here, unlike FD.  Row layout: ``W[:n1] = E_r`` (V1 dofs),
``W[n1:] = E_phi`` (constrained V0 dofs); ``V[:n0p] = h_r`` (V0), ``V[n0p:] =
h_phi`` (V1) -- V mirrors the FD ``(h_r, h_phi)`` ordering, each h component
represented on its flux-partner grid.  Per-layer meshes couple through
``sem_interface_smatrix``, which Galerkin-projects layer A's traces onto
layer B's spaces (exact quadrature on merged breakpoints) and then applies
the standard pointwise interface algebra -- the cylindrical analog of the
Cartesian PMM ``layer_grids="per-layer"``.

LITERATURE NOTE (design review 2026-08-26).  The provably-exact-sequence
formulation for the axisymmetric Fourier-mode de Rham complex works in the
1-form COORDINATE component ``u = r E_phi`` (Oh, arXiv:1911.08983; Stock &
Oh, SIURO 14; Copeland-Gopalakrishnan-Oh, Math. Comp. 79), with the axis
constraint ``u'(0) + m a(0) = 0`` (equivalently ``E_phi(0) = i m E_r(0)``);
Gerritsma edge functions then realize d/dr : V0 -> V1 as an incidence matrix.
This module instead discretizes the PHYSICAL component ``E_phi`` (matching
the FD basis and the cascade contract) and imposes only the integrability
conditions above.  The literature-predicted risk of that choice -- kernel
pollution / spurious modes for m != 0 -- is EMPIRICALLY EXCLUDED here:
zero interlopers among the top-25 modes at m = 1, 2 with p up to 16
(max rel err 7e-15 -- machine precision, no conditioning drift), and the
open-cladding fiber window contains exactly the oracle's modes
(tests/unit/test_bor_sem.py).  If future high-m work ever shows spurious
modes, the ``u = r E_phi`` substitution is the pinned fallback.  Related
precedent: Lee-Sun-Cendes (IEEE T-MTT 39, 1991) for the q^2 formulation;
Zschiedrich et al. (JCMmode) and Simsek (arXiv:2604.12014) for the
edge/nodal pairing; Boffi et al. (SINUM 44 / arXiv:0909.5079) for p-version
discrete compactness; Bonnet-Ben Dhia-Chesnel-Ciarlet (Comm. PDE 39) for the
sign-changing-eps (metal) well-posedness contrast criterion (Cu against LC
sits at kappa ~ -40, far from the critical interval).
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lu_factor

from .coupled_radial_eigensolver import _fast_geig
from .radial_eigensolver import _gll_nodes_weights, _lagrange_vals_derivs

__all__ = ["SemRadialMesh", "sem_layer_modes", "sem_interface_smatrix",
           "equalize_meshes"]

_C = complex


def _as_tri(eps):
    """Per-element eps -> (eps_rr, eps_phiphi, eps_zz) triple (isotropic
    scalars broadcast; same acceptance rule as ``_split_eps``)."""
    a = np.asarray(eps, dtype=complex).ravel()
    if a.size == 1:
        return (complex(a[0]),) * 3
    if a.size == 3:
        return (complex(a[0]), complex(a[1]), complex(a[2]))
    raise ValueError(
        "SEM element eps must be a scalar or the 3 diagonal cylindrical "
        f"components (eps_rr, eps_phiphi, eps_zz); got {a.size} value(s).")


class SemRadialMesh:
    """One layer's radial element mesh: breakpoints (element boundaries, ring
    walls INCLUDED, spanning exactly [0, Rbig]) + per-element eps triples +
    the shared polynomial degree ``p``."""

    def __init__(self, breakpoints, eps_elems, degree, *, nq_extra=8,
                 R_pml=None, sigma_max=5.0, pml_p=2):
        b = np.asarray(breakpoints, dtype=float)
        if b.ndim != 1 or b.size < 2:
            raise ValueError("SemRadialMesh: need >= 2 breakpoints")
        if abs(b[0]) > 0.0:
            raise ValueError("SemRadialMesh: first breakpoint must be r = 0")
        if np.any(np.diff(b) <= 0.0):
            raise ValueError(
                "SemRadialMesh: breakpoints must be strictly increasing")
        degree = int(degree)
        if degree < 2:
            raise ValueError("SemRadialMesh: degree must be >= 2")
        self.b = b
        self.Rbig = float(b[-1])
        # radial PML (M3): complex coordinate stretch r -> r + i INT sigma dr
        # for r > R_pml, sigma(r) = sigma_max ((r - R_pml)/(Rbig - R_pml))^p.
        # Assembly-level only: operators in the PML elements use the
        # stretched radius and 1/s d/dr; the outer wall stays PEC behind it.
        if R_pml is not None:
            R_pml = float(R_pml)
            if not (0.0 < R_pml < self.Rbig):
                raise ValueError(
                    f"SemRadialMesh: R_pml must lie in (0, Rbig), got "
                    f"{R_pml} vs Rbig={self.Rbig}.")
            if not np.any(np.abs(b - R_pml) < 1e-12 * self.Rbig):
                raise ValueError(
                    "SemRadialMesh: R_pml must coincide with an element "
                    "boundary (add it to the breakpoints).")
        self.R_pml = R_pml
        self.sigma_max = float(sigma_max)
        self.pml_p = int(pml_p)
        self.ne = b.size - 1
        self.p = degree
        self.eps = [_as_tri(e) for e in eps_elems]
        if len(self.eps) != self.ne:
            raise ValueError(
                f"SemRadialMesh: {self.ne} elements but {len(self.eps)} eps "
                "entries")
        # Axis-element anisotropy guard (Bhardwaj et al., C. R. Physique 21
        # (2020): the cylindrical frame is singular at r = 0, so a tensor
        # with eps_rr != eps_phiphi is NOT a well-defined field on the axis).
        er0, ep0, _ez0 = self.eps[0]
        if er0 != ep0:
            warnings.warn(
                "SemRadialMesh: the axis-containing element carries "
                f"eps_rr={er0} != eps_phiphi={ep0}; transverse anisotropy is "
                "undefined AT r = 0 (the cylindrical frame is singular "
                "there).  Keep the axis element transversely isotropic, or "
                "add an isotropic core element.", stacklevel=2)
        p = degree
        self.ref0, self.w0_ref = _gll_nodes_weights(p)   # V0 ref (p+1 nodes)
        g, wg = leggauss(p)                              # V1 ref (p nodes)
        self.ref1, self.w1_ref = g, wg
        self.n0 = self.ne * p + 1                        # V0 global dofs
        self.n1 = self.ne * p                            # V1 global dofs
        # assembly quadrature (over-integrated Gauss; interior points only)
        xq, wq = leggauss(p + int(nq_extra))
        self.xq, self.wq = xq, wq
        self.V0q, self.D0q = _lagrange_vals_derivs(self.ref0, xq)
        self.V1q, _unused = _lagrange_vals_derivs(self.ref1, xq)
        # V0 -> V1-node evaluation (h_phi needs E_z' at the Gauss nodes)
        self.V0g, self.D0g = _lagrange_vals_derivs(self.ref0, self.ref1)

    # -- dof maps ---------------------------------------------------------
    def g0(self, e):
        return np.arange(e * self.p, e * self.p + self.p + 1)

    def g1(self, e):
        return np.arange(e * self.p, (e + 1) * self.p)

    def el(self, e):
        a, bb = self.b[e], self.b[e + 1]
        return a, bb, 0.5 * (bb - a)

    # -- node radii + native diagonal weights (flux representation) -------
    def nodes0(self):
        r = np.empty(self.n0)
        w = np.zeros(self.n0)
        for e in range(self.ne):
            a, _bb, J = self.el(e)
            rr = a + (self.ref0 + 1.0) * J
            i0 = self.g0(e)
            r[i0] = rr
            w[i0] += self.w0_ref * J * rr                # lumped GLL r dr
        return r, w

    def nodes1(self):
        r = np.empty(self.n1)
        w = np.empty(self.n1)
        for e in range(self.ne):
            a, _bb, J = self.el(e)
            i1 = self.g1(e)
            r[i1] = a + (self.ref1 + 1.0) * J
            w[i1] = self.w1_ref * J * r[i1]              # exact Gauss r dr
        return r, w

    def stretch(self, r):
        """Complex PML map: returns (r_tilde, s) at radii ``r`` --
        the stretched radius and the stretch factor s = 1 + i sigma(r)."""
        r = np.asarray(r, float)
        if self.R_pml is None:
            return r.astype(complex), np.ones_like(r, dtype=complex)
        d = self.Rbig - self.R_pml
        x = np.clip((r - self.R_pml) / d, 0.0, None)
        sig = self.sigma_max * x ** self.pml_p
        # analytic integral of sigma: sigma_max d x^(p+1) / (p+1)
        acc = self.sigma_max * d * x ** (self.pml_p + 1) / (self.pml_p + 1)
        return r + 1j * acc, 1.0 + 1j * sig

    def fingerprint(self):
        return ("sem", self.p, self.b.tobytes(), tuple(self.eps),
                self.R_pml, self.sigma_max, self.pml_p)

    def quad_eval(self, *, nq_extra=None):
        """Oversampled per-element quadrature grid + dof->sample evaluation
        matrices, for OSCILLATORY projections (the far-field Fourier-Bessel
        kernel; cf. the Cartesian PMM's oversampled ``_sem_projection_quad``).
        Returns ``(rq, wq, E0, E1)``: sample radii, r-dr quadrature weights,
        and dense evaluators mapping V0 / V1 dof vectors to sample values."""
        nq = self.p + (2 * self.p + 8 if nq_extra is None else int(nq_extra))
        xq, wq_ref = leggauss(nq)
        V0e, _d0 = _lagrange_vals_derivs(self.ref0, xq)
        V1e, _d1 = _lagrange_vals_derivs(self.ref1, xq)
        rq = np.empty(self.ne * nq)
        wq = np.empty(self.ne * nq)
        E0 = np.zeros((self.ne * nq, self.n0))
        E1 = np.zeros((self.ne * nq, self.n1))
        for e in range(self.ne):
            a, _bb, J = self.el(e)
            sl = slice(e * nq, (e + 1) * nq)
            rq[sl] = a + (xq + 1.0) * J
            wq[sl] = wq_ref * J * rq[sl]
            E0[sl, self.g0(e)] = V0e
            E1[sl, self.g1(e)] = V1e
        return rq, wq, E0, E1


def _assemble(mesh, m, k0):
    """All Galerkin blocks (module-docstring formulas)."""
    n0, n1 = mesh.n0, mesh.n1

    def zeros(a, b):
        return np.zeros((a, b), dtype=_C)

    M_rr, N_rr, Krr_eps = zeros(n1, n1), zeros(n1, n1), zeros(n1, n1)
    M_pp, S_A, Kpp_eps = zeros(n0, n0), zeros(n0, n0), zeros(n0, n0)
    S_z, Mz_eps = zeros(n0, n0), zeros(n0, n0)
    C1, D_zr, Q = zeros(n1, n0), zeros(n0, n1), zeros(n0, n0)
    for e in range(mesh.ne):
        a, _bb, J = mesh.el(e)
        r_real = a + (mesh.xq + 1.0) * J
        r, s_pml = mesh.stretch(r_real)          # identity when no PML
        w = mesh.wq * J * s_pml                  # dr -> s dr
        er, ep, ez = mesh.eps[e]
        V0 = mesh.V0q
        D0 = mesh.D0q / J / s_pml[:, None]       # d/dr -> (1/s) d/dr
        V1 = mesh.V1q
        A0 = D0 + V0 / r[:, None]
        i0, i1 = mesh.g0(e), mesh.g1(e)
        wr = w * r
        m00 = (V0 * wr[:, None]).T @ V0
        m11 = (V1 * wr[:, None]).T @ V1
        M_rr[np.ix_(i1, i1)] += m11
        Krr_eps[np.ix_(i1, i1)] += er * m11
        N_rr[np.ix_(i1, i1)] += (V1 * (w / r)[:, None]).T @ V1
        M_pp[np.ix_(i0, i0)] += m00
        Kpp_eps[np.ix_(i0, i0)] += ep * m00
        S_A[np.ix_(i0, i0)] += (A0 * wr[:, None]).T @ A0
        S_z[np.ix_(i0, i0)] += (D0 * wr[:, None]).T @ D0 \
            + (m * m) * ((V0 * (w / r)[:, None]).T @ V0)
        Mz_eps[np.ix_(i0, i0)] += ez * m00
        C1[np.ix_(i1, i0)] += (V1 * w[:, None]).T @ A0
        D_zr[np.ix_(i0, i1)] += (D0 * wr[:, None]).T @ V1
        Q[np.ix_(i0, i0)] += (V0 * w[:, None]).T @ V0
    return dict(M_rr=M_rr, N_rr=N_rr, Krr_eps=Krr_eps, M_pp=M_pp, S_A=S_A,
                Kpp_eps=Kpp_eps, S_z=S_z, Mz_eps=Mz_eps, C1=C1, D_zr=D_zr,
                Q=Q)


def _keeps(mesh, m):
    """Essential-BC keep masks for the phi and z V0 spaces (PEC wall + axis).
    Both depend only on ``m`` (the wall is always PEC here), so every layer
    of a stack shares them -- the equal-dof invariant the cascade needs."""
    kp = np.ones(mesh.n0, bool)
    kz = np.ones(mesh.n0, bool)
    kp[-1] = False
    kz[-1] = False
    if m != 1:
        kp[0] = False
    if m != 0:
        kz[0] = False
    return np.where(kp)[0], np.where(kz)[0]


def sem_layer_modes(mesh, m, k0):
    """Solve one layer's radial modes on ``mesh``; return the cascade dict
    (module docstring: contract + row layout)."""
    m = int(m)
    k0 = float(k0)
    ops = _assemble(mesh, m, k0)
    ip, iz = _keeps(mesh, m)
    n1 = mesh.n1
    Mz = (-ops["S_z"] + k0 * k0 * ops["Mz_eps"])[np.ix_(iz, iz)]
    # near-singular E_z-elimination guard (sibling of the FD audit-P3-13
    # warning) -- but with a REMEDY the FD path lacks: at a longitudinal
    # resonance the Schur elimination is undefined (the q = 0 E_z mode makes
    # Mz singular), so fall back to the UNREDUCED pencil
    #     [[K, 0], [0, 0]] x = q^2 [[G, Cc], [-J, Mz]] x
    # solved by QZ, which needs no Mz inverse (the literature route: keep E_z
    # and accept a singular B -- Zschiedrich/JCMmode, Simsek).  Verified by
    # elimination: Cc Mz^{-1} J == B - G block-by-block.
    _du = np.abs(np.diag(lu_factor(Mz)[0]))
    _resonant = bool(_du.size and _du.min() <= 1e-12 * _du.max())
    if _resonant:
        warnings.warn(
            "sem_layer_modes: the E_z-elimination operator is near-singular "
            f"(LU pivot ratio {_du.min() / _du.max():.2e}) -- k0={k0:.10g} "
            "sits at a longitudinal resonance of this mesh; solving the "
            "UNREDUCED (E_r, E_phi, E_z) pencil by QZ instead of the Schur "
            "elimination.", stacklevel=2)
        Lei = None
    else:
        Lei = np.linalg.inv(Mz)
    Dzr = ops["D_zr"][iz, :]
    Qzp = ops["Q"][np.ix_(iz, ip)]
    K = np.block([
        [k0 * k0 * ops["Krr_eps"] - m * m * ops["N_rr"],
         -1j * m * ops["C1"][:, ip]],
        [+1j * m * ops["C1"][:, ip].T,
         (k0 * k0 * ops["Kpp_eps"] - ops["S_A"])[np.ix_(ip, ip)]]])
    if Lei is not None:
        B = np.block([
            [ops["M_rr"] + Dzr.T @ Lei @ Dzr, -1j * m * (Dzr.T @ Lei @ Qzp)],
            [+1j * m * (Qzp.T @ Lei @ Dzr),
             ops["M_pp"][np.ix_(ip, ip)] + m * m * (Qzp.T @ Lei @ Qzp)]])
        q2, X = _fast_geig(K, B)
    else:
        # unreduced pencil: x = (E_r, E_phi, zeta); constraint row makes the
        # non-physical directions infinite eigenvalues, filtered below
        from scipy.linalg import eig as _qz
        nzz = iz.size
        npsi = n1 + ip.size
        G = np.block([[ops["M_rr"], np.zeros((n1, ip.size), dtype=complex)],
                      [np.zeros((ip.size, n1), dtype=complex),
                       ops["M_pp"][np.ix_(ip, ip)]]])
        Cc = np.vstack([Dzr.T, +1j * m * Qzp.T])          # (npsi, nzz)
        J = np.hstack([Dzr, -1j * m * Qzp])               # (nzz, npsi)
        A_big = np.block([[K, np.zeros((npsi, nzz), dtype=complex)],
                          [np.zeros((nzz, npsi), dtype=complex),
                           np.zeros((nzz, nzz), dtype=complex)]])
        B_big = np.block([[G, Cc], [-J, Mz]])
        q2_all, X_all = _qz(A_big, B_big)
        fin = np.isfinite(q2_all) & (np.abs(q2_all) < 1e12 * k0 * k0)
        # keep the npsi finite eigenpairs (the physical block dimension)
        idx = np.where(fin)[0]
        idx = idx[np.argsort(-np.abs(q2_all[idx].real))][:npsi]
        q2 = q2_all[idx]
        X = X_all[:npsi, idx]
        _zeta = X_all[npsi:, idx]     # Mz zeta = J Psi -> Ez_red = -i zeta
    
    q = np.sqrt(np.asarray(q2, dtype=complex))
    nm = q.size
    # -- field recovery at the native nodes (nodal bases: value == dof) ----
    r0, w0 = mesh.nodes0()
    r1, w1 = mesh.nodes1()
    Er = X[:n1, :]
    Ephi_full = np.zeros((mesh.n0, nm), dtype=_C)
    Ephi_full[ip, :] = X[n1:, :]
    Ez_full = np.zeros((mesh.n0, nm), dtype=_C)      # E_z PER UNIT q
    if Lei is not None:
        Ez_full[iz, :] = Lei @ (-1j * (Dzr @ Er) - m * (Qzp @ X[n1:, :]))
    else:
        Ez_full[iz, :] = -1j * _zeta                 # from the pencil block
    # E_z / r at V0 nodes; axis node by the r^{|m|} limit (Ez ~ r for m != 0)
    Ez_over_r = np.zeros_like(Ez_full)
    pos = r0 > 0.0
    Ez_over_r[pos, :] = Ez_full[pos, :] / r0[pos, None]
    if m != 0 and not pos[0]:
        _a0, _b0, J0 = mesh.el(0)
        Daxis = _lagrange_vals_derivs(mesh.ref0, mesh.ref0[:1])[1] / J0
        Ez_over_r[0, :] = (Daxis @ Ez_full[mesh.g0(0), :])[0, :]
    # E_z' at the V1 (Gauss) nodes, per element
    dEz_g = np.zeros((n1, nm), dtype=_C)
    for e in range(mesh.ne):
        _a, _b, J = mesh.el(e)
        dEz_g[mesh.g1(e), :] = (mesh.D0g / J) @ Ez_full[mesh.g0(e), :]

    def hfields(qv):
        """h = sqrt(mu0/eps0) H, same normalization as the FD basis:
        h_r = (m Ez/r - q Ephi)/k0 on V0; h_phi = (q Er + i Ez')/k0 on V1."""
        hr = (m * (Ez_over_r * qv[None, :]) - qv[None, :] * Ephi_full) / k0
        hphi = (qv[None, :] * Er + 1j * (dEz_g * qv[None, :])) / k0
        return hr, hphi

    def zflux(hr, hphi):
        return np.real(
            np.sum(Er * np.conj(hphi) * w1[:, None], axis=0)
            - np.sum(Ephi_full * np.conj(hr) * w0[:, None], axis=0))

    hr, hphi = hfields(q)
    flux = zflux(hr, hphi)
    # forward orientation -- the zcascade branch rule, vectorized
    prop = np.abs(q.imag) < 1e-9 * np.maximum(np.abs(q.real), 1e-300)
    flip = np.where(prop, flux < 0.0, q.imag < 0.0)
    q = np.where(flip, -q, q)
    hr, hphi = hfields(q)
    flux = zflux(hr, hphi)
    # unit-flux normalization (audit-P1-01 relative threshold)
    fnrm = (np.sum(np.abs(Er) ** 2 * w1[:, None], axis=0)
            + np.sum(np.abs(Ephi_full) ** 2 * w0[:, None], axis=0))
    s = np.where(np.abs(flux) > 1e-10 * fnrm,
                 1.0 / np.sqrt(np.abs(flux) + 1e-300),
                 1.0 / (np.sqrt(fnrm) + 1e-300))
    Er = Er * s[None, :]
    Ephi_s = Ephi_full * s[None, :]
    hr = hr * s[None, :]
    hphi = hphi * s[None, :]
    W = np.vstack([Er, Ephi_s[ip, :]])
    V = np.vstack([hr[ip, :], hphi])                 # (h_r, h_phi) FD order
    return dict(W=W, V=V, q=q, n1=n1, n0p=ip.size, r=r0[ip], wq=w0[ip],
                r_face=r1, wq_face=w1, wq_node=w0[ip], N=n1, mesh=mesh,
                ip=ip, iz=iz)


def _overlap(mesh_a, mesh_b, which):
    """Exact Galerkin ``r dr`` overlap ``<phi_b_i, phi_a_j>`` between two
    meshes' same-kind spaces ("v0" or "v1"), by Gauss quadrature on the
    MERGED breakpoints (each merged interval lies inside ONE element of each
    mesh, so the integrand is a polynomial of degree <= 2p + 1 -- exact)."""
    nga = mesh_a.n0 if which == "v0" else mesh_a.n1
    ngb = mesh_b.n0 if which == "v0" else mesh_b.n1
    G = np.zeros((ngb, nga))
    cuts = np.unique(np.concatenate([mesh_a.b, mesh_b.b]))
    xq, wq = leggauss(max(mesh_a.p, mesh_b.p) + 2)
    ea = eb = 0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (lo + hi)
        while mesh_a.b[ea + 1] < mid:
            ea += 1
        while mesh_b.b[eb + 1] < mid:
            eb += 1
        J = 0.5 * (hi - lo)
        r = lo + (xq + 1.0) * J
        w = wq * J * r
        xa = 2.0 * (r - mesh_a.b[ea]) / (mesh_a.b[ea + 1] - mesh_a.b[ea]) - 1.0
        xb = 2.0 * (r - mesh_b.b[eb]) / (mesh_b.b[eb + 1] - mesh_b.b[eb]) - 1.0
        if which == "v0":
            Va = _lagrange_vals_derivs(mesh_a.ref0, xa)[0]
            Vb = _lagrange_vals_derivs(mesh_b.ref0, xb)[0]
            ia, ib = mesh_a.g0(ea), mesh_b.g0(eb)
        else:
            Va = _lagrange_vals_derivs(mesh_a.ref1, xa)[0]
            Vb = _lagrange_vals_derivs(mesh_b.ref1, xb)[0]
            ia, ib = mesh_a.g1(ea), mesh_b.g1(eb)
        G[np.ix_(ib, ia)] += (Vb * w[:, None]).T @ Va
    return G


def sem_interface_smatrix(La, Lb):
    """Interface S-matrix between two SEM layers on (possibly) DIFFERENT
    meshes -- the per-layer-grid transfer function.

    CROSS-TESTED Galerkin mode matching: tangential-E continuity is tested in
    layer B's spaces, tangential-H continuity in layer A's (the classic
    reaction-preserving mode-matching pairing).  Testing BOTH conditions in
    one layer's space makes the part of the other layer's rough (evanescent)
    mode traces orthogonal to that space leak into spurious reflection --
    measured |S11| ~ 5e1 on a same-medium cross-mesh round trip; the
    cross-tested form restores it to spectral accuracy.  With
    ``alpha = Wb^-1 P_ba Wa`` (E, tested in B) and
    ``gamma = Va^-1 P_ab Vb`` (H, tested in A)::

        S11 = (I + gamma alpha)^-1 (I - gamma alpha)
        S12 = 2 (I + gamma alpha)^-1 gamma
        S21 = alpha (I + S11)
        S22 = alpha S12 - I

    which reduces ALGEBRAICALLY to ``zcascade.interface_smatrix`` when the
    meshes coincide (P = I: gamma alpha = b^-1 a, so S11 = (a+b)^-1 (b-a),
    S12 = 2 (a+b)^-1, ...)."""
    from .zcascade import interface_smatrix
    ma, mb = La["mesh"], Lb["mesh"]
    n1a = La["n1"]
    same = (ma.p == mb.p and ma.b.shape == mb.b.shape
            and np.array_equal(ma.b, mb.b))
    if same:
        return interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])
    if La["n0p"] != Lb["n0p"] or n1a != Lb["n1"]:
        raise ValueError(
            "sem_interface_smatrix: adjacent layers must carry equal dof "
            "counts (run the meshes through `equalize_meshes`); got "
            f"(n1, n0p) = ({n1a}, {La['n0p']}) vs "
            f"({Lb['n1']}, {Lb['n0p']}).")
    ipa, ipb = La["ip"], Lb["ip"]
    # A -> B projections (E tested in B)
    G1_ba = _overlap(ma, mb, "v1")
    G0_ba = _overlap(ma, mb, "v0")
    P1_ba = np.linalg.solve(_overlap(mb, mb, "v1"), G1_ba)
    M0b = _overlap(mb, mb, "v0")[np.ix_(ipb, ipb)]
    P0_ba = np.linalg.solve(M0b, G0_ba[np.ix_(ipb, ipa)])
    # B -> A projections (H tested in A); overlap transposes swap the roles
    P1_ab = np.linalg.solve(_overlap(ma, ma, "v1"), G1_ba.T)
    M0a = _overlap(ma, ma, "v0")[np.ix_(ipa, ipa)]
    P0_ab = np.linalg.solve(M0a, G0_ba[np.ix_(ipb, ipa)].T)
    Wa_b = np.vstack([P1_ba @ La["W"][:n1a], P0_ba @ La["W"][n1a:]])
    Vb_a = np.vstack([P0_ab @ Lb["V"][:Lb["n0p"]], P1_ab @ Lb["V"][Lb["n0p"]:]])
    alpha = np.linalg.solve(Lb["W"], Wa_b)
    gamma = np.linalg.solve(La["V"], Vb_a)
    n = alpha.shape[0]
    eye = np.eye(n, dtype=complex)
    ga = gamma @ alpha
    inv_iga = np.linalg.inv(eye + ga)
    S11 = inv_iga @ (eye - ga)
    S12 = 2.0 * (inv_iga @ gamma)
    S21 = alpha @ (eye + S11)
    S22 = alpha @ S12 - eye
    return (S11, S12, S21, S22)


def equalize_meshes(meshes):
    """Pad every mesh (splitting its largest element at the midpoint) until
    all carry the same element count -- the equal-dof invariant the square
    interface algebra needs.  Deterministic; returns NEW meshes."""
    target = max(msh.ne for msh in meshes)
    out = []
    for msh in meshes:
        b = list(msh.b)
        eps = list(msh.eps)
        while len(b) - 1 < target:
            widths = np.diff(np.asarray(b))
            j = int(np.argmax(widths))
            b.insert(j + 1, b[j] + widths[j] / 2.0)
            eps.insert(j, eps[j])
        out.append(SemRadialMesh(b, eps, msh.p))
    return out
