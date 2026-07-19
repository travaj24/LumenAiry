"""Canonical no-floor 2-D crossed-grating PMM -- Granet 2023 staggered
modified-Legendre basis.
================================================================================

A NON-Fourier rigorous modal solver for a 2-D (doubly periodic) crossed grating
of axis-aligned rectangular pillars -- the 2-D analogue of
:func:`~lumenairy.elements.pmm.pmm_efficiency_1d`, and the *no-floor* counterpart
of the FMM-floored hybrid :func:`~lumenairy.elements.pmm.twod.pmm_efficiency_2d`.

This implements the FAITHFUL staggered modified-Legendre basis of Granet,
"Modal spectral element method with modified Legendre polynomials to analyze
binary crossed gratings," J. Opt. Soc. Am. A 40, 652 (2023) (Eqs. 23-34) -- NOT
a DG-broken-space + flux-penalty substitute.  Continuity is EMBEDDED in the
basis via shared hats (Eq. 32) and the Bloch periodic hat (Eq. 33); there is NO
stabilization parameter, so there is NO mechanism to inject spurious modes.  The
two staggered 1-D sets ``Btilde`` (continuous, C0) and ``B`` (its discontinuous
partner, equal cardinality) realize the 2-D Li inverse-rule placement BY
CONSTRUCTION: each transverse field component is continuous across the wall it
crosses and reduced (bubble) in the other direction.  The longitudinal field is
slaved by ``div(D) = 0`` (Eq. 16-18; the ``-K_tz (eps33)^-1 K_zt`` Schur term in
``L``), which removes the longitudinal spurious sea.

Why this over the FMM-floored hybrid
-------------------------------------
* **No Fourier floor.**  Every region (cover, film, substrate) is solved in the
  SAME staggered modal basis at the same dimension, so every interface is a
  SQUARE modal match (the 1-D PMM architecture lifted to 2-D); the
  Rayleigh-order projection is applied ONCE, FORWARD only, at the far field.
  The energy balance is therefore ``n_orders``-INDEPENDENT and tracks ONLY the
  modal degree ``M`` (verified to ~1e-13 machine precision).  The hybrid, by
  contrast, projects the layer into a truncated Fourier basis before the
  eigensolve and inherits the FMM ``n_orders`` floor.
* **Exact sidewalls + position invariance.**  The pillar walls land on element
  boundaries (Eq. 26), so ``eps`` is exact per element (no Gibbs); the total
  efficiencies are invariant to the pillar's position in the cell.

Scope / limitations
-------------------
* **Axis-aligned RECTANGULAR pillars only** -- the walls must coincide with the
  segment boundaries of the ``(Nx, Ny)`` ``eps_cell`` grid (Eq. 26).  Curved /
  slanted boundaries need Granet's transfinite curved-quad mapping (not
  implemented).
* **Corner-capped.**  A right-angle dielectric pillar has field singularities at
  its four corners, so the bound-mode (and hence efficiency) convergence is
  ALGEBRAIC, not spectral -- monotone with NO floor, but at-best RCWA-parity
  per DOF on vertical pillars (the universal corner cap; see
  :func:`~lumenairy.elements.pmm.grating_convergence_class`).  The genuine win is
  accuracy QUALITY (no-floor, exact sidewalls, position invariance, pinning the
  value RCWA converges toward), not raw per-DOF speed.  A smooth-field region
  (vacuum / homogeneous) converges spectrally.
* Single layer, isotropic scalar TE/TM, NumPy/SciPy dense generalized eig (not
  JAX-differentiable).

Conventions match the rest of the library: PUBLIC ``exp(-i w t)`` (``n = n + i
kappa``, ``Im eps > 0`` for loss), forward ``exp(+i kz z)``, ``Im(kz) >= 0``.
Eigenvalue ``gamma^2/k0^2 = n_eff^2``; derivatives are ``(1/k0) d/dx``.

Equations implemented (Granet 2023, verified against the paper):
  Eq.23-24 :  -gamma^2 R [E1;E2] = L [E1;E2];  R = C[chi_t]C (nonmagnetic
              chi_t=I -> R=C@C=-I);  L = k^2[eps_t] + S_tt - K_tz(eps33)^-1 K_zt
  Eq.20-22 :  S_tt = [d2;-d1] chi33 [d2,-d1];  K_tz = C[d2;-d1];
              K_zt = [d1 eps, d2 eps] (isotropic)
  Eq.16-18 :  E3 slaved by div(D)=0 (the Schur term)
  Eq.30-33 :  modified Legendre Ltilde_m; continuity hats; Bloch periodic hat
  Eq.34    :  staggered tensor expansion E1=B(x)Btilde, E2=Btilde(x)B,
              E3=Btilde(x)Btilde
  Eq.39-44 :  Appendix-A bra-ket matrix elements (real inner product INT x* y).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

from ..rcwa import Efficiency2D  # cross-suite 2-D result (unpacks (o,R,T), carries .dof)
from ..rcwa._core import _project_efficiency  # shared flux-projection (S1-9/S1-10)

# UNCHANGED S-matrix algebra from the shipped 1-D PMM -- every region's modal
# matrix W is square (same staggered dimension), so these are plain square
# solves (no pseudo-inverse, no weighting).
from ._core import (
    _forward_branch_flip,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)

__all__ = ["pmm_efficiency_2d_staggered"]

_C = np.complex128


# === 1-D modified-Legendre staggered basis + 2-D transverse-E eigensolver ===
# (Granet 2023 Eqs.30-34 / 23-24; faithful no-shortcuts staggered basis)
def _legendre_value_deriv(maxdeg, u):
    """L_k(u) and L_k'(u) for k=0..maxdeg at scalar/array u, via recurrence.
    Returns arrays of shape (maxdeg+1,) + u.shape."""
    u = np.asarray(u, dtype=float)
    P = np.zeros((maxdeg + 1,) + u.shape)
    dP = np.zeros((maxdeg + 1,) + u.shape)
    P[0] = 1.0
    if maxdeg >= 1:
        P[1] = u
        dP[1] = 1.0
    for k in range(1, maxdeg):
        P[k + 1] = ((2 * k + 1) * u * P[k] - k * P[k - 1]) / (k + 1)
        dP[k + 1] = ((2 * k + 1) * (P[k] + u * dP[k]) - k * dP[k - 1]) / (k + 1)
    return P, dP


def _modleg_value_deriv(M, u):
    """Modified-Legendre Ltilde_m(u), m=1..M, and derivative, on u in [-1,1].
    Eq.30:  m=1 -> (L0-L1)/2,  m=2 -> (L0+L1)/2,  m>2 -> L_{m-1...} L_m - L_{m-2}.
    We index the returned array 0..M-1 for m=1..M.  For m>2 (array index >=2)
    the polynomial is L_{idx} - L_{idx-2} where idx = m-1 ... careful indexing:
    Granet's m runs 1,2,3,...  L_m is the Legendre poly of degree m.  So
    m=1 uses L0,L1; m=2 uses L0,L1; m=3 -> L_3 - L_1? NO.  Re-read Eq.30:
        Ltilde_m = L_m - L_{m-2} for m>2.
    With m the Legendre DEGREE label, m=3 -> L_3 - L_1, m=4 -> L_4 - L_2, etc.
    But that makes Ltilde_3, Ltilde_4 NOT vanish at the endpoints?  L_m(+/-1) =
    (+/-1)^m, so L_m(1)-L_{m-2}(1) = 1-1 = 0 and L_m(-1)-L_{m-2}(-1) =
    (-1)^m-(-1)^{m-2} = 0.  YES -- they vanish at both ends (bubbles).  Good.
    So array index a=0 -> m=1, a=1 -> m=2, a>=2 -> m=a+1 (bubble L_{a+1}-L_{a-1}).
    """
    maxdeg = max(2, M)        # need up to L_M
    Pp, dPp = _legendre_value_deriv(maxdeg, u)  # values+derivs
    vals = []
    ders = []
    for a in range(M):
        if a == 0:                       # Ltilde_1 = (L0 - L1)/2  (left half-hat)
            vals.append(0.5 * (Pp[0] - Pp[1]))
            ders.append(0.5 * (dPp[0] - dPp[1]))
        elif a == 1:                     # Ltilde_2 = (L0 + L1)/2  (right half-hat)
            vals.append(0.5 * (Pp[0] + Pp[1]))
            ders.append(0.5 * (dPp[0] + dPp[1]))
        else:                            # bubble: L_a - L_{a-2} (degrees 2,3,...,M-1)
            # array index a (>=2) -> degree-a bubble L_a - L_{a-2}, vanishing at
            # both endpoints (L_a(+/-1)=(+/-1)^a, so L_a-L_{a-2}=0 at +/-1).
            # This INCLUDES the degree-2 bubble (a=2: L_2-L_0) -- the previous
            # L_{a+1}-L_{a-1} reading SKIPPED degree 2 and left the C0 space
            # incomplete (wrong plane-wave eigenvalues for N>2).
            vals.append(Pp[a] - Pp[a - 2])
            ders.append(dPp[a] - dPp[a - 2])
    return np.array(vals), np.array(ders)


class Basis1D:
    """One-period 1-D staggered modified-Legendre basis on [0, d], N segments,
    M modified-Legendre functions per segment.  Builds BOTH global sets:

      Btilde : continuous C0 set (hats glued across nodes + Bloch periodic hat).
               dimension  Ntil = N*(M-1).
      B      : staggered partner with one bubble dropped per interval, SAME dim.

    Each global function is represented by its expansion in the per-segment local
    modified-Legendre functions: a global function g has, on segment n, a vector
    of M local coefficients g_local[n, 0:M] (coeff of Ltilde_a^n).  This local-
    coefficient representation lets us assemble ALL Galerkin integrals exactly
    from the per-segment elementary matrices of the modified-Legendre functions
    (computed once by Gauss-Legendre quadrature of high order -> exact for
    polynomials).
    """

    def __init__(self, d, N, M, tau=1.0 + 0.0j):
        assert M >= 3, "Basis1D needs M>=3 (M=2 gives a degenerate cardinality)"
        self.d = float(d)
        self.N = int(N)
        self.M = int(M)
        self.tau = _C(tau)
        self.h = self.d / self.N                 # segment length
        self.J = 0.5 * self.h                    # dx/du jacobian
        # segment boundaries on the eps walls (Eq.31, uniform)
        self.xb = np.linspace(0.0, self.d, self.N + 1)
        self._build_elementary()
        self._build_sets()

    # ----- per-segment elementary modified-Legendre matrices (reference u) ---
    def _build_elementary(self):
        M = self.M
        # high-order Gauss-Legendre: exact for products up to degree ~2M
        gx, gw = leggauss(2 * M + 4)
        V, Vp = _modleg_value_deriv(M, gx)       # (M, Q)
        # mass  m_ab = INT_{-1}^{1} Ltilde_a Ltilde_b du
        self.m_ref = (V * gw) @ V.T               # (M, M)
        # stiffness s_ab = INT Ltilde_a' Ltilde_b' du
        self.s_ref = (Vp * gw) @ Vp.T
        # mixed  c_ab = INT Ltilde_a' Ltilde_b du   (for derivative-bra terms)
        self.c_ref = (Vp * gw) @ V.T
        # endpoint values (for hat continuity / inspection)
        Ve, _ = _modleg_value_deriv(M, np.array([-1.0, 1.0]))
        self.val_m1 = Ve[:, 0]                    # Ltilde_a(-1)
        self.val_p1 = Ve[:, 1]                    # Ltilde_a(+1)

    # ----- assemble the two global sets as local-coefficient stencils --------
    def _build_sets(self):
        """Each global function -> a list of (segment, M-vector) contributions.
        We store, for each global dof j, a dense (N, M) complex coefficient
        array S[j] giving its local modified-Legendre coefficients on every
        segment.  Sparse in practice (few nonzero segments) but small here."""
        N, M, tau = self.N, self.M, self.tau

        # local modified-Legendre indices:  0 -> Ltilde_1 (left half-hat),
        # 1 -> Ltilde_2 (right half-hat),  2..M-1 -> bubbles.
        # ---- Btilde : N hats (periodic) + bubbles 2..M-1 on every segment ----
        til = []          # list of (N,M) arrays
        # hats T^n, n=1..N (Eq.32 / Eq.33).  T^n glues Ltilde_2 of seg n-1 to
        # Ltilde_1 of seg n (0-based: node between seg n-1 and seg n).  The
        # periodic hat T (the one spanning the period seam) uses tau.
        for node in range(N):                 # node index 0..N-1 (N nodes, periodic)
            S = np.zeros((N, M), dtype=_C)
            left_seg = (node - 1) % N          # segment ending at this node
            right_seg = node                   # segment starting at this node
            if node == 0:
                # Bloch periodic hat (Eq.33): Ltilde_1 on seg 0, tau*Ltilde_2 on seg N-1
                S[right_seg, 0] += 1.0
                S[left_seg, 1] += tau
            else:
                S[left_seg, 1] += 1.0          # Ltilde_2 on the left segment
                S[right_seg, 0] += 1.0         # Ltilde_1 on the right segment
            til.append(S)
        # interior bubbles: one global dof per (segment, bubble index 2..M-1)
        for seg in range(N):
            for a in range(2, M):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                til.append(S)
        self.Btilde = til                       # dimension N + N*(M-2) = N*(M-1)

        # ---- B : staggered partner, SAME dimension, one bubble dropped/interval
        # We build B as the DISCONTINUOUS set: per segment, BOTH half-hats as
        # INDEPENDENT functions (NOT glued) + bubbles 2..M-2 (drop the LAST
        # bubble a=M-1 per segment).  Per segment that is 2 + (M-3) = M-1
        # functions -> total N*(M-1), matching |Btilde| exactly.
        b = []
        for seg in range(N):
            # two independent half-hats on this segment
            for a in (0, 1):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
            for a in range(2, M - 1):           # bubbles, drop last (a=M-1)
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
        self.B = b
        assert len(self.Btilde) == len(self.B) == N * (M - 1), \
            (len(self.Btilde), len(self.B), N * (M - 1))
        self.dim = N * (M - 1)

    # ----- global Galerkin matrices between two sets -------------------------
    def _global_matrix(self, ref, setL, setR, eps_seg=None):
        """<setL_i | (ref) | setR_j> on the period, assembled from per-segment
        elementary matrix `ref` (one of m_ref/s_ref/c_ref on reference u).
        Physical scaling: mass-type (m_ref) -> *J ; stiffness s_ref -> *(1/J);
        mixed c_ref (one derivative) -> *1 (the du cancels the 1/J of d/dx times
        the J of du).  `eps_seg` (length N) multiplies the per-segment integral
        (piecewise-constant eps; element walls on eps steps -> exact).
        Real inner product INT x* y -> conjugate the LEFT coefficients."""
        N, _M = self.N, self.M
        if ref is self.m_ref:
            scale = self.J
        elif ref is self.s_ref:
            scale = 1.0 / self.J
        else:                                    # c_ref : one derivative
            scale = 1.0
        # Stack each set into a (dim, N, M) tensor for vectorized contraction.
        L_ten = np.array(setL)                    # (dimL, N, M)
        R_ten = np.array(setR)                    # (dimR, N, M)
        w_seg = np.ones(N, dtype=_C) * scale
        if eps_seg is not None:
            w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
        # out[i,j] = sum_seg w_seg[seg] * conj(L[i,seg,:]) @ ref @ R[j,seg,:]
        # = sum_seg w_seg[seg] * (conj(L[i,seg]) . (ref @ R[j,seg]))
        RR = np.einsum("ab,jsb->jsa", ref, R_ten)         # ref @ R[j,seg]  (dimR,N,M)
        out = np.einsum("isa,s,jsa->ij",
                        np.conj(L_ten), w_seg, RR)
        return out

    # convenience builders (mass=m_ref, stiff=s_ref, mixed=c_ref) ------------
    def mass(self, setL, setR, eps_seg=None):
        return self._global_matrix(self.m_ref, setL, setR, eps_seg)

    def stiff(self, setL, setR, eps_seg=None):
        return self._global_matrix(self.s_ref, setL, setR, eps_seg)

    def mixed(self, setL, setR, eps_seg=None):
        """<setL | d/dx | setR> : INT setL* (d setR/dx).  c_ref[a,b] =
        INT Ltilde_a'(u) Ltilde_b(u) du has the derivative on the LEFT index, so
        for INT L_i*(setR)' we need c_ref with derivative on the RIGHT operand ->
        use c_ref.T contracted appropriately: build <L | R'> = conj(ci) @ c_ref.T
        @ cj.  We pass ref=c_ref.T-like by transposing in the contraction."""
        L_ten = np.array(setL)
        R_ten = np.array(setR)
        # INT setL* d(setR)/dx du-physical: d/dx = (1/J) d/du, du = ... the J and
        # 1/J cancel -> scale 1.  c_ref[a,b]=INT Ltilde_a' Ltilde_b; we want
        # INT (setL)_a (setR)_b' = c_ref[b,a] -> use c_ref.T.
        cT = self.c_ref.T
        RR = np.einsum("ab,jsb->jsa", cT, R_ten)
        out = np.einsum("isa,jsa->ij", np.conj(L_ten), RR)
        return out


# =========================================================================== #
# 2-D STAGGERED TENSOR ASSEMBLY  (Eq.34) + transverse-E eigenproblem (Eq.23-24)
# --------------------------------------------------------------------------- #
# Field spaces (tensor products of the two 1-D sets):
#   V1 (E1) = B(x1)      (x) Btilde(x2)
#   V2 (E2) = Btilde(x1) (x) B(x2)
#   V3 (E3) = Btilde(x1) (x) Btilde(x2)
# Each per-axis set has dimension  q = N*(M-1).  So dim V1=dim V2=dim V3 = q^2.
#
# Operators (k = 2*pi numerically; coords in units of wavelength; we report
# g2 = gamma^2/k0^2 = -lam/k0^2 where lam is the raw eigenvalue of the
# generalized pencil).  We build everything with PHYSICAL d/dx (units 1/lambda).
#
# Per-axis elementary global matrices we need, between the relevant set pairs:
#   Mtt   = <Btilde | Btilde>            (mass, x-axis or y-axis)
#   Mtt_e = <Btilde | eps | Btilde>      (eps-weighted mass)
#   Mbb   = <B | B>,   Mbb_e = <B|eps|B>
#   Cbt   = <B | d/dx Btilde>            (mixed: bubble-test, hat-trial deriv)
#   Ctb   = <Btilde | d/dx B>
#   Ctt   = <Btilde | d/dx Btilde>
#   Stt   = <Btilde'| Btilde'> etc as needed
# With separable eps(x1,x2)=eps1(x1)*... NO: eps is a 2-D map (pillar).  For a
# rectangular pillar eps(x1,x2) is SEPARABLE on the element grid ONLY if the
# pillar is the full-width product of an x-interval and a y-interval -> YES, a
# rectangular pillar IS separable: eps = eps_p inside [xa,xb]x[ya,yb].  We handle
# the general (separable-in-segments) case by per-(seg_x,seg_y) constant eps and
# a Kronecker assembly: the 2-D eps-weighted mass is sum over (sx,sy) of
# eps[sx,sy] * (x-seg-mass kron y-seg-mass).  For a rectangular pillar this is
# exact (walls on element boundaries, Eq.26).
# =========================================================================== #


def _global_pair_segmat(basis: Basis1D, ref, setL, setR):
    """Like _global_matrix but returns the PER-SEGMENT contributions stacked:
    G[seg] (dimL,dimR) so the 2-D eps-weighted assembly can weight each
    (segx,segy) cell.  ref in {m_ref}; scale handled here."""
    _N, _M = basis.N, basis.M
    if ref is basis.m_ref:
        scale = basis.J
    elif ref is basis.s_ref:
        scale = 1.0 / basis.J
    else:
        scale = 1.0
    L_ten = np.array(setL)        # (dimL,N,M)
    R_ten = np.array(setR)        # (dimR,N,M)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)            # (dimR,N,M)
    # per-segment matrix G[s,i,j] = scale * conj(L[i,s]) . RR[j,s]
    G = scale * np.einsum("isa,jsa->sij", np.conj(L_ten), RR)
    return G                       # (N, dimL, dimR)


class Granet2DTransverseE:
    """Faithful Granet staggered transverse-E eigensolver for a rectangular
    (or separable) 2-D unit cell, isotropic nonmagnetic media.

    Parameters
    ----------
    px, py    : period (units of wavelength)
    Nx, Ny    : segments per axis (walls on eps steps)
    M         : modified-Legendre functions per segment per axis (degree knob)
    eps_cell  : (Nx, Ny) array of constant eps per segment-cell
    alpha0x, alpha0y : Bloch wavenumbers (units 1/lambda) -> tau_x, tau_y
    k0        : 2*pi / wavelength (numerically 2*pi if coords in wavelengths)
    """

    def __init__(self, px, py, Nx, Ny, M, eps_cell,
                 alpha0x=0.0, alpha0y=0.0, k0=2.0 * np.pi):
        self.k0 = float(k0)
        self.alpha0x = float(alpha0x)
        self.alpha0y = float(alpha0y)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1D(px, Nx, M, taux)
        self.by = Basis1D(py, Ny, M, tauy)
        self.eps_cell = np.asarray(eps_cell, dtype=_C)   # (Nx, Ny)
        self.q = self.bx.dim                              # = Nx*(M-1)
        assert self.bx.dim == self.by.dim, "use square (Nx*(M-1)==Ny*(M-1))"
        self._assemble()

    # --- per-axis 1-D ingredient matrices between set pairs (no eps) ---------
    def _axis_mats(self):
        bx, by = self.bx, self.by
        # x-axis
        self.Mtt_x = bx.mass(bx.Btilde, bx.Btilde)        # <til|til>
        self.Mbb_x = bx.mass(bx.B, bx.B)                  # <B|B>
        self.Ctb_x = bx.mixed(bx.Btilde, bx.B)            # <til| d B>
        self.Cbt_x = bx.mixed(bx.B, bx.Btilde)            # <B  | d til>
        self.Ctt_x = bx.mixed(bx.Btilde, bx.Btilde)       # <til| d til>
        # y-axis
        self.Mtt_y = by.mass(by.Btilde, by.Btilde)
        self.Mbb_y = by.mass(by.B, by.B)
        self.Ctb_y = by.mixed(by.Btilde, by.B)
        self.Cbt_y = by.mixed(by.B, by.Btilde)
        self.Ctt_y = by.mixed(by.Btilde, by.Btilde)

    # --- eps-weighted per-axis-pair segment tensors (for 2-D eps assembly) ---
    def _eps_weighted(self, refx_pair, refy_pair):
        """Assemble a 2-D eps-weighted Galerkin matrix:
          sum_{sx,sy} eps[sx,sy] * kron( Gy[sy], Gx[sx] )
        where Gx = per-segment x-matrix between the x set-pair, Gy similarly.
        refx_pair = (basis_x, refmat_x, setLx, setRx); same for y."""
        bx, refx, sLx, sRx = refx_pair
        by, refy, sLy, sRy = refy_pair
        Gx = _global_pair_segmat(bx, refx, sLx, sRx)      # (Nx, dLx, dRx)
        Gy = _global_pair_segmat(by, refy, sLy, sRy)      # (Ny, dLy, dRy)
        eps = self.eps_cell                               # (Nx, Ny)
        dLx, dRx = Gx.shape[1], Gx.shape[2]
        dLy, dRy = Gy.shape[1], Gy.shape[2]
        out = np.zeros((dLy * dLx, dRy * dRx), dtype=_C)
        for sx in range(bx.N):
            # accumulate over sy with eps weights -> weighted y-matrix
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)   # (dLy,dRy)
            out += np.kron(Wy, Gx[sx])
        return out

    def _assemble(self):
        bx, by = self.bx, self.by
        k0 = self.k0
        self._axis_mats()
        q = self.q
        qq = q * q

        # ----- per-axis 1-D primitive matrices (real inner product) -----
        # Mass (no deriv) between set pairs:
        # Only the like-set masses enter the staggered Gram (G1, G2 below).
        Mtt_x = self.Mtt_x      # <til|til>_x
        Mtt_y = self.Mtt_y
        Mbb_x = self.Mbb_x      # <B|B>_x
        Mbb_y = self.Mbb_y
        # Directed weak derivative (deriv on the TRIAL/right operand), /k0 ->
        # dimensionless.  d[L,R] = INT L* (d R/dx) /k0.  Only <B | d til> enters
        # the K_zt / K_tz coupling terms below.
        dbt_x = bx.mixed(bx.B, bx.Btilde) / k0        # <B  | d til>_x
        dbt_y = by.mixed(by.B, by.Btilde) / k0

        # ============ R = C[chi_t]C = -I  -> -block Gram (mass) ===============
        G1 = np.kron(Mtt_y, Mbb_x)                  # <V1|V1>, V1=B(x1)til(x2)
        G2 = np.kron(Mbb_y, Mtt_x)                  # <V2|V2>, V2=til(x1)B(x2)
        Rmat = np.zeros((2 * qq, 2 * qq), dtype=_C)
        Rmat[:qq, :qq] = -G1
        Rmat[qq:, qq:] = -G2

        # ============ L = [eps_t] + S_tt - K_tz eps33^-1 K_zt  (/k0^2) ========
        # --- [eps_t] : eps-weighted component masses (k^2[eps]/k0^2 = [eps]) ---
        Et_11 = self._eps_weighted(
            (bx, bx.m_ref, bx.B, bx.B),
            (by, by.m_ref, by.Btilde, by.Btilde))   # E1 space, V1
        Et_22 = self._eps_weighted(
            (bx, bx.m_ref, bx.Btilde, bx.Btilde),
            (by, by.m_ref, by.B, by.B))             # E2 space, V2

        # --- DIRECTED DERIVATIVE OPERATORS into the V3 (Btil x Btil) space ---
        # We discretize the STRONG symbol exactly as in the analytic check:
        #   curl-like  w = d2 E1 - d1 E2   (V3 coeffs via the V3 Gram)
        #   div-like   s = d1(eps E1) + d2(eps E2)
        # and form S_tt, Schur as G3-mediated products of these directed ops.
        #
        # V3 Gram (no eps), the E3=Btil(x)Btil space, would be
        # kron(Mtt_y, Mtt_x) -- never consumed (the curl is tested in its own
        # Vw space below, and the Schur solve runs against Meps33), so it is
        # not built (audit P3-37; already flagged 2026-06-08).

        # === S_tt : the curl chi33 curl operator ===
        # MIMETIC PLACEMENT (the crux of the staggered basis).  The de Rham
        # property d(Btilde) subset span(B) (verified ~1e-15) means the CURL
        #   w = d2 E1 - d1 E2,  E1=B(x1)Btil(x2), E2=Btil(x1)B(x2)
        # lands EXACTLY in the space  Vw = B(x1) (x) B(x2)  (NOT V3=Btil(x)Btil):
        #   d2 E1 = B(x1) Btil'(x2) ,  Btil'(x2) in span(B) -> B(x1)(x)B(x2)
        #   d1 E2 = Btil'(x1) B(x2) ,  Btil'(x1) in span(B) -> B(x1)(x)B(x2)
        # Testing the curl in its OWN natural space Vw=B(x)B (Gram Gw) is what
        # makes the discrete curl FULL-RANK on the transverse modes and renders
        # the de Rham complex EXACT -> spurious-free.  (Projecting the curl into
        # V3=Btil(x)Btil instead -- the previous attempt -- loses rank because
        # B' is NOT in span(Btil), leaving a longitudinal residue = spurious sea.)
        Gw = np.kron(Mbb_y, Mbb_x)                  # <Vw|Vw>, Vw=B(x1)(x)B(x2)
        # Curl (Vw <- [E1;E2]):
        #   <w| d2 E1> : x1 <B|B>=Mbb_x ; x2 <B| d Btil>=dbt_y -> kron(dbt_y, Mbb_x)
        Cw_E1 = np.kron(dbt_y, Mbb_x)
        #   <w| d1 E2> : x1 <B| d Btil>=dbt_x ; x2 <B|B>=Mbb_y -> kron(Mbb_y, dbt_x)
        Cw_E2 = np.kron(Mbb_y, dbt_x)
        Curl = np.concatenate([Cw_E1, -Cw_E2], axis=1)     # (q^2, 2q^2)
        Gw_inv = np.linalg.inv(Gw)
        # IBP: <t,[d2;-d1] w> = -<Curl(t), w>  -> S_tt = -Curl^dag Gw^{-1} Curl.
        Stt = -Curl.conj().T @ Gw_inv @ Curl        # (2q^2,2q^2), neg-semidef

        # === K_tz eps33^-1 K_zt : the div(D)=0 Schur term (Eq.16-18) ===
        # MIMETIC GRADIENT (the dual de Rham map).  K_tz = C[chi_t][d2;-d1] acts
        # V3 -> [E1;E2]; with chi_t=I, C[d2;-d1]=[-d1;-d2], i.e. K_tz V3 =
        #   [ -d1 V3 ; -d2 V3 ].  Since d Btil in span(B):
        #     d1 V3 = (d1 Btil)(x1) Btil(x2) in B(x1)(x)Btil(x2) = V1
        #     d2 V3 = Btil(x1) (d2 Btil)(x2) in Btil(x1)(x)B(x2) = V2
        #   so the GRADIENT maps V3 cleanly INTO [V1;V2] -- FULL RANK (the dual of
        #   the curl's exactness).  We build K_tz as the Galerkin gradient and
        #   K_zt as its adjoint through the component masses, with eps inserted in
        #   the middle (eps33^-1 on the V3 side; [eps_t] on the field side is
        #   already in L's first term -- here K_zt carries the eps of the D-field
        #   divergence d(eps E)).
        #
        # Build the eps-free mimetic gradient blocks (V1<-V3, V2<-V3):
        #   <t1 | -d1 V3> : t1 in V1=B(x1)Btil(x2), V3=Btil(x1)Btil(x2)
        #     x1: <B | d Btil> = dbt_x ; x2: <Btil|Btil> = Mtt_y
        Grad1 = -np.kron(Mtt_y, dbt_x)              # V1 <- V3   (=<t1|-d1 V3>)
        Grad2 = -np.kron(dbt_y, Mtt_x)              # V2 <- V3   (=<t2|-d2 V3>)
        Ktz = np.concatenate([Grad1, Grad2], axis=0)       # (2q^2, q^2)

        # eps33 mass in V3 (Eq.41): <V3| eps | V3>
        Meps33 = self._eps_weighted(
            (bx, bx.m_ref, bx.Btilde, bx.Btilde),
            (by, by.m_ref, by.Btilde, by.Btilde))

        # K_zt = [d1 eps, d2 eps] (V3 <- [E1;E2]).  This is the divergence of the
        # D-field d(eps E).  The CONSISTENT (mimetic) discretization is the eps-
        # weighted negative adjoint of the gradient: <V3| d1(eps E1)> assembled so
        # that, with eps piecewise-constant on the cells (walls on element bnds),
        # it equals  -<(d1 V3) , eps E1>  with the SAME component masses as the
        # gradient.  We assemble it via the eps-weighted component overlap of the
        # gradient blocks: K_zt = -(eps-weighted Ktz-partner)^dag is NOT generally
        # right when eps varies, so we assemble K_zt DIRECTLY in V3 with eps:
        #   <V3| d1(eps E1)> = -<d1 V3 | eps E1> = -eps-weighted <(d Btil)B>_x ...
        #   per cell: x1 -< (d Btil) | eps | B >_x ; x2 <Btil|eps|Btil>_y
        #   (deriv-on-LEFT, i.e. on the V3 test) -- realized by _eps_dir with the
        #   derivative on the TEST set Btilde of the x-axis.
        Kzt_E1 = -self._eps_dir(bx, "Btilde", "dL", "B",
                                by, "Btilde", "m", "Btilde") / k0   # <V3|d1(eps)|V1>
        Kzt_E2 = -self._eps_dir(bx, "Btilde", "m", "Btilde",
                                by, "Btilde", "dL", "B") / k0       # <V3|d2(eps)|V2>
        Kzt = np.concatenate([Kzt_E1, Kzt_E2], axis=1)     # (q^2, 2q^2)

        # Schur term  K_tz @ Meps33^{-1} @ K_zt  (eps33^-1 = solve vs Meps33).
        Schur = Ktz @ np.linalg.solve(Meps33, Kzt)

        # ----- assemble L -----
        Lmat = np.zeros((2 * qq, 2 * qq), dtype=_C)
        Lmat[:qq, :qq] = Et_11
        Lmat[qq:, qq:] = Et_22
        Lmat += Stt
        Lmat -= Schur

        # Only the operators DOWNSTREAM consumers read are retained
        # (_region_modes: Lmat/Rmat/Et_blocks/Stt; _homog_geom_cache adds
        # Schur).  Curl/Kzt/Ktz/Meps33 stay assembly locals -- storing them
        # retained ~44% more dead operator memory per solver at convergence-
        # grade sizes (audit P3-37; e.g. 1.66 GB per solver at Nx=Ny=4, M=16).
        self.Rmat = Rmat
        self.Lmat = Lmat
        self.Et_blocks = (Et_11, Et_22)
        self.Stt = Stt
        self.Schur = Schur
        self.dimtot = 2 * qq

    # --- eps-weighted directed (deriv) 2-D operator into V3 ------------------
    def _eps_dir(self, bx, lx, opx, rx, by, ly, opy, ry):
        """Assemble  sum_{sx,sy} eps[sx,sy] * kron(Gy, Gx)  where Gx is the
        per-segment x-matrix between sets (lx-test, rx-trial) with op opx in
        {'m'(mass), 'd'(deriv-on-trial)}, similarly Gy.  Realizes the eps-
        weighted <V3 | d_k (eps .) | Vt> matrix exactly (walls on cell bnds)."""
        def segmat(basis, lset, op, rset):
            sL = getattr(basis, lset)
            sR = getattr(basis, rset)
            Lt = np.array(sL)
            Rt = np.array(sR)
            if op == "m":                           # mass (no deriv)
                RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
                scale = basis.J
            elif op == "dL":                        # deriv on LEFT (test): INT (dL) R
                # c_ref[a,b]=INT La' Lb -> INT (dL_a) R_b = conj(L)@c_ref@R
                RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
                scale = 1.0
            else:                                   # 'd' deriv on trial (right): INT L (dR)
                RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
                scale = 1.0
            return scale * np.einsum("isa,jsa->sij", np.conj(Lt), RR)
        Gx = segmat(bx, lx, opx, rx)
        Gy = segmat(by, ly, opy, ry)
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", self.eps_cell[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out



# === full canonical solve: square all-SEM match + Li S-matrix + far field ===
# =========================================================================== #
# decay-branch helpers (public exp(-i w t))
# =========================================================================== #
def _sqrt_decay(x):
    r = np.sqrt(np.asarray(x, dtype=_C))
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


def _inv_lam(lam):
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _kz_forward2(eps, kx, ky):
    val = np.sqrt(np.asarray(eps - kx ** 2 - ky ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


# =========================================================================== #
# Fourier projection of the staggered modified-Legendre 1-D set onto Rayleigh
# orders:   T[m, j] = (1/d) INT_0^d  phi_j(x) exp(-i m G x) dx,   G = 2pi/d.
# A global staggered dof j has local modified-Legendre coefficients S[j][seg, a]
# (the same (N,M) stencil the eigensolver stores).  We integrate each local
# modified-Legendre function exactly by Gauss-Legendre quadrature on its segment.
# This is the staggered-basis analogue of pmm._sem_fourier_projection.
# =========================================================================== #
def _stag_fourier_projection(basis: Basis1D, orders, alpha0=0.0):
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    nq = 2 * M + 8
    xg, wg = leggauss(nq)                       # on [-1,1]
    Vref, _ = _modleg_value_deriv(M, xg)        # (M, nq) modified-Legendre values
    orders = np.asarray(orders)
    # per-segment contribution of local function a to Rayleigh order m.  The
    # modal field is a BLOCH mode: the tau-glued basis carries the transverse
    # momentum (tau = exp(-i alpha0 d)), so the physical field is
    # ~ exp(-i alpha0 x) p(x), and the library's Rayleigh order m (far-field
    # kxv = kx0 + m*wl/period) has transverse dependence e^{-i(alpha0 + mG)x}.
    # Its amplitude is recovered by projecting on the CONJUGATE kernel
    # e^{+i(mG + alpha0) x}.  Both pieces of the kernel matter and each failure
    # is invisible in a subset of tests: omitting ``alpha0`` breaks the
    # one-period orthogonality at oblique (energy loss even in the specular
    # order); flipping the sign of ``m`` deposits physical order -m into slot m
    # (a MIRROR) -- exact at normal incidence (kz even in kx) and for
    # reflection-symmetric cells, but at oblique every |m|>0 order gets the
    # WRONG kz flux factor (kz(+m) applied to order -m's amplitude), which
    # leaked/gained several % energy for patterned cells (asymmetric-cell
    # per-order oracle test vs pmm_efficiency_2d_cell pinned the form
    # stag[m] = oracle[-m]*kz(m)/kz(-m) before this fix).
    #   c[m, seg, a] = (1/d) INT_seg Ltilde_a(u(x)) e^{+i(mG + alpha0)x} dx
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    J = basis.J
    for seg in range(N):
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg     # physical x at quad pts
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))   # (nO, nq)
        # contribution[m,a] = (J/d) sum_q phase[m,q] wg[q] Vref[a,q]
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T
    # assemble onto the global dofs of a chosen set (list of (N,M) stencils)
    def _assemble(global_set):
        S = np.array(global_set)               # (dim, N, M)
        # T[m, j] = sum_{seg,a} T_local[m,seg,a] * S[j,seg,a]
        return np.einsum("msa,jsa->mj", T_local, S)
    return _assemble


def _far_projector_2d(bx: Basis1D, by: Basis1D, ox, oy, alpha0x=0.0, alpha0y=0.0):
    """Forward Fourier->Rayleigh projectors for the 2-D staggered field
    components.  Returns the per-component (E1,E2) projection operators that map
    a region's [E1;E2] modal coefficient vector onto the Rayleigh orders.

    E1 = sum B(x1) Btilde(x2)  -> project x1 with B-set, x2 with Btilde-set.
    E2 = sum Btilde(x1) B(x2)  -> project x1 with Btilde-set, x2 with B-set.
    Tensor ordering matches the eigensolver's kron(y, x): index I = jx + qx*jy.

    ``alpha0x``/``alpha0y`` are the per-axis Bloch wavenumbers (``kx0*k0`` /
    ``ky0*k0``); the projection kernel carries them so the once-only far-field
    Rayleigh projection stays ORTHOGONAL at oblique incidence (see
    :func:`_stag_fourier_projection`).  Default 0 = normal incidence (byte-
    identical to the un-shifted projection).
    """
    asmx = _stag_fourier_projection(bx, ox, alpha0x)
    asmy = _stag_fourier_projection(by, oy, alpha0y)
    Tx_B = asmx(bx.B)            # (Mx, qx)
    Tx_til = asmx(bx.Btilde)    # (Mx, qx)
    Ty_B = asmy(by.B)            # (My, qy)
    Ty_til = asmy(by.Btilde)    # (My, qy)
    # E1 in V1 = B(x1) (x) Btilde(x2):  P1 = kron(Ty_til, Tx_B)
    P1 = np.kron(Ty_til, Tx_B)   # (Mx*My, qx*qy)
    # E2 in V2 = Btilde(x1) (x) B(x2):  P2 = kron(Ty_B, Tx_til)
    P2 = np.kron(Ty_B, Tx_til)
    return P1, P2


def _pmm2d_project_orders(P1, P2, Wmodes, qq):
    """Project a PMM-2D ``[E1; E2]`` modal matrix onto the Rayleigh orders --
    the ``_proj`` closure formerly duplicated in :mod:`.stack2d_pure` and this
    module (audit S1-10).  ``P1``/``P2`` map the E1 (Ex) / E2 (Ey) nodal blocks
    (``qq`` rows each) onto the orders; returns the stacked
    ``[Ex_orders; Ey_orders]``.  Reproduces the former closure operation-for-
    operation."""
    top = P1 @ Wmodes[:qq, :]
    bot = P2 @ Wmodes[qq:, :]
    return np.concatenate([top, bot], axis=0)


def _pmm2d_order_kz(eps_sup, eps_sub, kxv, kyv, kx0, ky0):
    """Per-order forward ``kz`` for the two half-spaces, the incident ``kz``, and
    the safe-divide ``kz`` used by the longitudinal-field reconstruction -- the
    ``kz_ref``/``kz_trn``/``kz_inc``/``safe_r``/``safe_t`` block formerly
    duplicated in :mod:`.stack2d_pure` and this module (audit S1-10).  Returns
    ``(kz_ref, kz_trn, kz_inc, safe_r, safe_t)``, reproducing the former inline
    block byte-for-byte."""
    kz_ref = _kz_forward2(eps_sup, kxv, kyv)
    kz_trn = _kz_forward2(eps_sub, kxv, kyv)
    kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    return kz_ref, kz_trn, kz_inc, safe_r, safe_t


# =========================================================================== #
# Region modes from the staggered eigensolver, with the Eq.25 H-partner.
# =========================================================================== #
def _region_modes(solver: Granet2DTransverseE):
    """Solve  L v = g2 G v  (G = -R, block field Gram), return:
        W    : tangential-E modal matrix [E1;E2] coefficients (2 q^2, 2 q^2)
        V    : tangential-H modal partner via Eq.25
        lam  : z-decay constants  sqrt(-g2)  (forward Im>=0)
        g2   : eigenvalues (gamma/k0)^2
    """
    L = solver.Lmat
    G = -solver.Rmat                  # block field Gram (Hermitian PD)
    g2, W = sla.eig(L, G)
    # q = kz/k0 = gamma/k0 = sqrt(g2).  FORWARD branch chosen ROBUSTLY (the
    # naive _sqrt_decay flips degenerate real-g2 pairs inconsistently on QZ
    # noise -> the H-partner sign flips and the S-matrix loses passivity).  We
    # pick Im(q) >= 0 (evanescent decay) with a noise tolerance, and for the
    # (near-)real propagating modes Re(q) > 0 (outgoing).  lam = -i q forward.
    q = np.sqrt(np.asarray(g2, dtype=_C))
    q = _forward_branch_flip(q)       # shared scalar-vertical selector (S1-8)
    lam = -1j * q                     # forward propagator exp(-lam k0 z) decays
    gamma_over_k0 = q

    # H recovery (Eq.25):  gamma C [H1;H2] = Lhh [E1;E2]
    #   Lhh = Et + Stt  (the L WITHOUT the div(D)=0 Schur term).
    #   [H1;H2] = (1/gamma) C^{-1} Lhh [E]  = (k0/(gamma/k0)) (-C) G^{-1} Lhh [E].
    Et11, Et22 = solver.Et_blocks
    qq = solver.q * solver.q
    Lhh = np.zeros_like(L)
    Lhh[:qq, :qq] = Et11
    Lhh[qq:, qq:] = Et22
    Lhh = Lhh + solver.Stt
    # block C^{-1} = -C = [[0,-1],[1,0]] acting on the 2-block coeff vector:
    #   (-C) [a;b] = [-b; a]  (a = top block, b = bottom block).
    Ginv = np.linalg.inv(G)
    Dual = Ginv @ (Lhh @ W)          # G^{-1} Lhh W  (back to coefficients)
    top = Dual[:qq, :]
    bot = Dual[qq:, :]
    rot = np.concatenate([-bot, top], axis=0)     # (-C) Dual
    inv_g = _inv_lam(gamma_over_k0)               # 1/(gamma/k0)
    V = rot * inv_g[None, :]
    return W, V, lam, g2


def _homog_geom_cache(solver: Granet2DTransverseE):
    """Pre-solve the eps-FREE geometric eig shared by EVERY homogeneous region.

    For a uniform-eps region the modal operator splits EXACTLY into an eps-scaled
    field-Gram plus an eps-free geometric part::

        L(eps) = eps * G + L0_geom ,   G = -Rmat = blockdiag(G1, G2)

    because (i) the component masses are ``Et_jj = eps * G_jj`` and (ii) the
    div(D)=0 Schur term is eps-INVARIANT -- its ``Meps33 = eps*G3`` and
    ``Kzt = eps*Kzt0`` cancel, so ``Schur = Ktz @ solve(G3, Kzt0)`` carries no eps
    (verified eps-independent to ~5e-29).  Hence ``L0_geom = Stt - Schur`` is purely
    geometric, and ONE generalized eig ``L0_geom W0 = g2_geo G W0`` serves all
    half-spaces: each region's modes are the SAME eigenvectors ``W0`` with the
    scalar-shifted spectrum ``g2 = g2_geo + eps`` (see :func:`_homog_region_modes`).
    This replaces the two homogeneous-region eigs (superstrate + substrate) with a
    single shared one -- 3 region eigs -> 2, the dominant cost (the eig is ~97% of a
    region solve).  ``solver`` must be a HOMOGENEOUS assembly (uniform ``eps_cell``)
    so its ``Stt``/``Schur``/``Rmat`` are the geometric operators.
    """
    G = -solver.Rmat                       # block field Gram (Hermitian PD)
    Stt = solver.Stt
    L0_geom = Stt - solver.Schur           # = Lmat - eps*G, manifestly eps-free
    g2_geo, W0 = sla.eig(L0_geom, G)
    Ginv = np.linalg.inv(G)
    qq = solver.q * solver.q
    # Pre-fold the eps-free pieces of the H-partner recovery (Eq.25): for a
    # homogeneous region Lhh = Et + Stt = eps*G + Stt, so Lhh @ W0 = eps*(G W0) +
    # (Stt W0) -- both terms eps-free and reusable across regions.
    return W0, g2_geo, G @ W0, Stt @ W0, Ginv, qq


def _homog_region_modes(geom, eps):
    """Modes of a homogeneous region (uniform permittivity ``eps``) from the shared
    eps-free geometric eig -- NO per-region eig.

    Mirrors :func:`_region_modes` EXACTLY (same forward-branch selection and Eq.25
    H-partner), but the eigenvectors are the cached ``W0`` and the spectrum is the
    scalar shift ``g2 = g2_geo + eps`` (the field-Gram term contributes ``+eps`` to
    every eigenvalue; the eigenvectors are unchanged).  The H-partner uses the
    pre-folded ``Lhh @ W0 = eps*(G W0) + (Stt W0)`` -- a cheap scalar combination, no
    matmul against a fresh operator.  Returns ``(W, V, lam)`` (g2 is not needed
    downstream for the half-spaces)."""
    W0, g2_geo, GW0, SttW0, Ginv, qq = geom
    g2 = g2_geo + eps
    q = np.sqrt(np.asarray(g2, dtype=_C))
    q = _forward_branch_flip(q)            # shared scalar-vertical selector (S1-8)
    lam = -1j * q                          # forward propagator exp(-lam k0 z) decays
    # H recovery (Eq.25) with Lhh = eps*G + Stt (homogeneous -> no Schur term):
    Dual = Ginv @ (eps * GW0 + SttW0)      # G^{-1} Lhh W0  (back to coefficients)
    top = Dual[:qq, :]
    bot = Dual[qq:, :]
    rot = np.concatenate([-bot, top], axis=0)     # (-C) Dual
    V = rot * _inv_lam(q)[None, :]
    return W0, V, lam


# =========================================================================== #
# THE FULL CANONICAL 2-D PMM SOLVE.
# =========================================================================== #
def pmm_efficiency_2d_staggered(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    degree: int = 8,
    n_modes: int | None = None,
    n_orders: int = 7,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
) -> Efficiency2D:
    """Rigorous diffraction efficiencies of a 2-D crossed grating of axis-aligned
    rectangular pillars by the canonical no-floor Polynomial Modal Method (Granet
    2023 staggered modified-Legendre basis).

    The NO-FLOOR 2-D counterpart of :func:`pmm_efficiency_1d` and of the
    FMM-floored hybrid :func:`~lumenairy.elements.pmm.twod.pmm_efficiency_2d`: the
    energy balance is ``n_orders``-INDEPENDENT (no Fourier floor) and tracks only
    the modal ``degree``.  See the module docstring for the method and its scope.

    Parameters
    ----------
    period_x, period_y : float
        Unit-cell periods along x and y (metres).
    eps_cell : (Nx, Ny) array_like of complex
        The layer permittivity as a constant value PER SEGMENT of the
        ``Nx`` x ``Ny`` rectangular grid.  The pillar walls are the segment
        boundaries, so ``eps`` is exact per element (Eq. 26) -- the grid must
        resolve the rectangular pillar exactly (e.g. a centred half-fill pillar
        is ``Nx = Ny = 2``).  The grid MUST be SQUARE (``Nx == Ny``; the
        staggered tensor-product basis requires ``Nx*(M-1) == Ny*(M-1)``) --
        tile a uniform axis into equal segments, or use
        :func:`pmm_efficiency_2d_cell`, which handles non-square cells.
        PUBLIC convention ``Im(eps) > 0`` for loss.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space indices (``n = n + i kappa``).
    depth : float
        Grating layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    degree : int, optional
        Number ``M`` of modified-Legendre functions per segment per axis -- the
        modal convergence knob (raise for accuracy; the per-component DOF is
        ``(Nx*(M-1)) * (Ny*(M-1))``).  Default 8.  NOTE (naming divergence): unlike
        the GLL POLYNOMIAL ``degree`` of :func:`pmm_efficiency_2d` and the 1-D
        ``pmm_*`` solvers, this ``degree`` is a BASIS-FUNCTION COUNT ``M`` -- so a
        cross-solver ``degree`` sweep compares unlike quantities.  Prefer the clearer
        alias ``n_modes`` (below); ``degree`` is kept for backward compatibility.
    n_modes : int, optional
        Clearer alias for ``degree`` (the modified-Legendre mode count ``M``).  If
        given, it overrides ``degree``.  Use this in new code.
    n_orders : int, optional
        Half-width of the retained Rayleigh diffraction-order set for the
        once-only forward far-field projection (the result is independent of
        this as long as it covers the propagating orders).  Default 7.
    polarization : {'te', 'tm'}, optional
        Incident polarization.  Default ``'te'``.
    theta, phi : float, optional
        Incidence polar / azimuthal angles (radians).  Default normal incidence.

    Returns
    -------
    result : :class:`~lumenairy.elements.rcwa.Efficiency2D`
        A ``tuple`` subclass that unpacks as ``orders, R, T`` (NOT a 4-tuple) and
        carries the modal problem size on ``.dof``:

        * ``orders`` -- ``(Nfo, 2)`` retained ``(m, n)`` diffraction-order pairs;
        * ``R, T`` -- ``(Nfo,)`` reflected / transmitted efficiency per order
          (real power fractions; evanescent orders 0; lossless ``sum(R)+sum(T)
          == 1``);
        * ``result.dof`` -- per-region modal problem size
          (``2 * Nx*(M-1) * Ny*(M-1)``).

        .. note:: **API change (v5.11 -> v5.12).**  Formerly a bare 4-tuple
           ``(orders, R, T, dof)``; now the cross-suite :class:`Efficiency2D`
           that unpacks to ``orders, R, T`` with ``dof`` as an attribute.  See
           :func:`pmm_efficiency_2d`.

    Notes
    -----
    Axis-aligned rectangular pillars only (walls on the ``eps_cell`` grid);
    corner-capped (algebraic, no-floor) convergence -- at-best RCWA parity per
    DOF on vertical pillars, the win being accuracy quality (no floor, exact
    sidewalls, position invariance).  NumPy/SciPy dense generalized eig.
    """
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_2d_staggered: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if n_modes is not None:
        # 'n_modes' is the clearer alias for the modified-Legendre mode count M --
        # 'degree' here is NOT a GLL polynomial degree (see the docstring); 'degree'
        # stays the default for backward compatibility.
        degree = int(n_modes)
    if int(degree) < 3:
        # M=2 passes the old >=2 guard but yields |Btilde|=N, |B|=2N and trips the
        # Basis1D cardinality assert deep in the build (audit P2); require M>=3.
        raise ValueError("pmm_efficiency_2d_staggered: degree (modified-Legendre "
                         "count M) must be >= 3.")
    eps_cell = np.asarray(eps_cell, dtype=_C)
    if eps_cell.ndim != 2:
        raise ValueError(
            f"pmm_efficiency_2d_staggered: eps_cell must be a 2-D (Nx, Ny) "
            f"array, got shape {eps_cell.shape}.")
    # SQUARE-grid restriction (audit P3-36): the staggered tensor-product
    # basis requires Nx*(M-1) == Ny*(M-1), i.e. Nx == Ny -- previously enforced
    # only by a deep assert (stripped under ``python -O`` into a cryptic
    # kron/broadcast shape error).  Fail loudly at the entry point instead.
    if eps_cell.shape[0] != eps_cell.shape[1]:
        raise ValueError(
            f"pmm_efficiency_2d_staggered: eps_cell must be SQUARE (Nx == Ny; "
            f"the staggered tensor-product basis requires "
            f"Nx*(M-1) == Ny*(M-1)), got shape {eps_cell.shape}.  Pad the "
            f"uniform axis into equal segments (e.g. tile a (2, 1) cell to "
            f"(2, 2)), or use pmm_efficiency_2d_cell, which handles "
            f"non-square cells.")
    polarization = pol
    Nx, Ny = eps_cell.shape
    M = int(degree)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2
    # Wood-anomaly guard (v5.14 robustness audit P1): the staggered solver's
    # H-partner ``V ~ 1/gamma`` amplifies basis-truncation error for a
    # JUST-PROPAGATING order, and the lossless total DIVERGES like
    # ~1/sqrt(cutoff distance) as the wavelength approaches any Rayleigh
    # cutoff FROM BELOW (measured tot = 1.2 at wl = P*(1 - 1e-6), 7.5 at
    # -1e-9, 205 at -1e-12).  Nudge off an EXACT coincidence, and WARN loudly
    # inside the divergence band (the hybrid pmm_efficiency_2d_cell is clean
    # there -- use it near cutoffs).
    from ..rcwa._core import (
        _grazing_safe_wavelength,
        _require_propagating_incidence,
    )
    nre0 = float(np.real(np.sqrt(eps_sup)))
    _mo = np.arange(-int(n_orders), int(n_orders) + 1)
    _mx = np.tile(_mo, len(_mo))
    _my = np.repeat(_mo, len(_mo))
    _kx0n = nre0 * np.sin(theta) * np.cos(phi)
    _ky0n = nre0 * np.sin(theta) * np.sin(phi)
    # Incidence guard (mirrors pmm_efficiency_2d / rcwa_efficiency_2d, v5.14.1
    # suite-wide fix): reject a gain superstrate (kz_inc forward root flips
    # negative -> every T silently negated) and an evanescent incident wave
    # (kz_inc ~ 0 divides the flux normalization).  This module keeps eps in
    # the PUBLIC exp(-iwt) convention (Im eps > 0 for loss), so conjugate into
    # the guard's INTERNAL convention.
    _require_propagating_incidence("pmm_efficiency_2d_staggered",
                                   np.conj(eps_sup),
                                   _kx0n ** 2 + _ky0n ** 2)
    wl = _grazing_safe_wavelength(float(wavelength), _kx0n, _ky0n, _mx, _my,
                                  period_x, period_y, [eps_sup, eps_sub])
    _kt2 = ((_kx0n + _mx * (wl / period_x)) ** 2
            + (_ky0n + _my * (wl / period_y)) ** 2)
    _gap = min(float(np.min(np.abs(float(np.real(e)) - _kt2)))
               for e in (eps_sup, eps_sub))
    if _gap < 1e-4:
        import warnings
        warnings.warn(
            f"pmm_efficiency_2d_staggered: a diffraction order is within "
            f"{_gap:.2g} (kt^2 units) of a Rayleigh cutoff; the staggered "
            f"solver's accuracy DEGRADES like ~1/sqrt(distance) near cutoffs "
            f"(energy errors of several % to >100% measured inside 1e-6).  "
            f"Use pmm_efficiency_2d_cell (clean there) or detune the "
            f"wavelength.", stacklevel=2)
    k0 = 2.0 * np.pi / wl
    nre = float(np.real(np.sqrt(eps_sup)))
    alpha0x = nre * np.sin(theta) * np.cos(phi) * k0
    alpha0y = nre * np.sin(theta) * np.sin(phi) * k0
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    # ---- region eigensolvers (all on the SAME grid/basis) ----
    sol_l = Granet2DTransverseE(period_x, period_y, Nx, Ny, M, eps_cell,
                                alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    Wl, Vl, lam_l, _g2l = _region_modes(sol_l)

    # The two HALF-SPACES are HOMOGENEOUS (uniform eps), where L(eps) = eps*G +
    # L0_geom with G, L0_geom eps-FREE.  So ONE eps-free generalized eig of the
    # geometric operator serves BOTH -- each region's modes are the SAME eigenvectors
    # with a scalar-shifted spectrum g2 = g2_geo + eps (no per-region eig).  This
    # turns the 3 region eigs into 2 (the eig is ~97% of a region solve) at machine-
    # identical R/T.  We assemble ONE homogeneous solver to source the geometric
    # G/Stt/Schur, then reconstruct both half-spaces.  See _homog_geom_cache.
    sol_h = Granet2DTransverseE(period_x, period_y, Nx, Ny, M,
                                np.full((Nx, Ny), _C(eps_sup)),
                                alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    geom = _homog_geom_cache(sol_h)
    # The geom tuple carries everything the half-spaces need; release the
    # homogeneous solver's large operator attributes before the S-matrix /
    # far-field stages (audit P3-37 -- keeping BOTH solver instances alive
    # through the far field retained several GB at convergence-grade M).
    del sol_h
    Wsup, Vsup, _ls = _homog_region_modes(geom, eps_sup)
    Wsub, Vsub, _lb = _homog_region_modes(geom, eps_sub)

    # ---- SQUARE Redheffer recursion (every W is 2 q^2) ----
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # ---- FORWARD-only far-field Fourier->Rayleigh projection (once) ----
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = len(order_x)
    P1, P2 = _far_projector_2d(sol_l.bx, sol_l.by, ox, oy,
                               alpha0x, alpha0y)             # (Nfo, q^2) each
    qq = sol_l.q * sol_l.q
    # H_E[m]-projector on a [E1;E2] modal matrix -> [Ex_orders; Ey_orders]:
    #   note: E1 = E_x component, E2 = E_y component (Granet transverse comps).
    Hsup = _pmm2d_project_orders(P1, P2, Wsup, qq)     # (2 Nfo, 2 q^2)
    Hsub = _pmm2d_project_orders(P1, P2, Wsub, qq)

    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)

    # incident (0,0) plane wave -> superstrate mode amplitudes by FORWARD overlap
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        axu, ayu = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ayu, axu
            einc_sq = 1.0
        else:
            ex0, ey0 = axu, ayu
            kz_inc0 = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    rhs = np.concatenate([ex0 * delta, ey0 * delta])         # (2 Nfo,)
    cinc, *_ = np.linalg.lstsq(Hsup, rhs, rcond=None)        # (2 q^2,)

    r_ord = Hsup @ (S11 @ cinc)                              # (2 Nfo,)
    t_ord = Hsub @ (S21 @ cinc)
    rx, ry = r_ord[:Nfo], r_ord[Nfo:]
    tx, ty = t_ord[:Nfo], t_ord[Nfo:]

    kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
        eps_sup, eps_sub, kxv, kyv, kx0, ky0)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R, T = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                               rx, ry, rz, tx, ty, tz, einc_sq)
    orders2d = np.stack([order_x, order_y], axis=1)
    # cross-suite return shape: unpacks as (orders, R, T); .dof = 2*q^2 (the modal
    # eigenproblem dimension).  Was a bare 4-tuple (orders, R, T, dof) pre-v5.12.
    return Efficiency2D(orders2d, R, T, 2 * qq)
