"""Stage-B prototype: OUT-OF-PLANE anisotropy in the PURE staggered 2-D PMM.

SELF-CONTAINED.  The staggered modified-Legendre basis (``Basis1D`` and its two
helper polynomial routines) is COPIED VERBATIM from
``lumenairy/elements/pmm/twod_staggered.py`` at f70628d so that this probe does
not move when the Stage-A build agent edits the library.  Everything else here
is NEW prototype code.  The library is imported ONLY for oracles and for
convention-free shared helpers (``_project_efficiency``,
``_select_forward_flux``, the Redheffer algebra).

CONVENTIONS (all PUBLIC, ``exp(-i w t)``, ``Im eps > 0`` = loss)
---------------------------------------------------------------
Fields:  ``F(x,y,z) = F(x,y) exp(+i gamma z)``, ``q = gamma/k0``.
Normalized derivative ``D_j = (1/k0) d/dx_j``;  ``D_3 -> i q``.
Normalized magnetic field ``G = i Z0 H`` makes Maxwell REAL-coefficient::

    D x E = G ,      D x G = eps E .

BLOCH SIGN (measured, see ``m0_convention.py``).  ``Basis1D``'s periodic hat
glues ``Ltilde_1`` on segment 0 to ``tau * Ltilde_2`` on segment N-1, so a basis
function obeys ``f(d) = tau f(0)``, i.e. it carries Bloch factor
``exp(+i K x)`` with ``tau = exp(+i K d)``.  This probe therefore passes
``tau = exp(+i k0 kx0 px)`` and the m-th harmonic is ``exp(+i(k0 kx0 + m G)x)``
-- the PHYSICAL harmonic, so ``D_1 -> +i kx/k0``.  (The shipped module passes
``tau = exp(-i alpha0 px)``, i.e. the opposite sign, which is UNOBSERVABLE
there because every derivative enters its operators an even number of times.
It is NOT unobservable once ``eps_xz`` couples one derivative to one field.)

Field spaces (Granet 2023 Eq. 34, unchanged):
    E1 in V1 = B(x) x Btil(y)      G2 = i Z0 Hy in V1
    E2 in V2 = Btil(x) x B(y)      G1 = i Z0 Hx in V2
    E3 in V3 = Btil(x) x Btil(y)   G3 = i Z0 Hz in Vw = B(x) x B(y)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

import lumenairy  # noqa: E402

# The worktree this probe belongs to, DERIVED from this file's location
# (validation/probe_pmm2d_staggered_oop/probe_common.py -> repo root) so the
# guard keeps biting when the probe is run from a different worktree: the
# hard-coded literal it replaces (C:/tmp/lum_aniso_oop) is a PREFIX of the
# Stage-B integration worktree C:/tmp/lum_aniso_oopint, so it would have
# passed there while asserting nothing.
WORKTREE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def assert_worktree():
    p = os.path.abspath(lumenairy.__file__)
    assert p.startswith(WORKTREE + os.sep), f"WRONG lumenairy: {p}"
    return p


assert_worktree()

_C = np.complex128

# =========================================================================== #
# COPIED VERBATIM from lumenairy/elements/pmm/twod_staggered.py (f70628d)
# =========================================================================== #


def _legendre_value_deriv(maxdeg, u):
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
    maxdeg = max(2, M)
    Pp, dPp = _legendre_value_deriv(maxdeg, u)
    vals, ders = [], []
    for a in range(M):
        if a == 0:
            vals.append(0.5 * (Pp[0] - Pp[1]))
            ders.append(0.5 * (dPp[0] - dPp[1]))
        elif a == 1:
            vals.append(0.5 * (Pp[0] + Pp[1]))
            ders.append(0.5 * (dPp[0] + dPp[1]))
        else:
            vals.append(Pp[a] - Pp[a - 2])
            ders.append(dPp[a] - dPp[a - 2])
    return np.array(vals), np.array(ders)


class Basis1D:
    """One-period 1-D staggered modified-Legendre basis (copy of the shipped
    class; ``tau`` is the Bloch glue factor ``f(d) = tau f(0)``)."""

    def __init__(self, d, N, M, tau=1.0 + 0.0j):
        assert M >= 3
        self.d = float(d)
        self.N = int(N)
        self.M = int(M)
        self.tau = _C(tau)
        self.h = self.d / self.N
        self.J = 0.5 * self.h
        self.xb = np.linspace(0.0, self.d, self.N + 1)
        self._build_elementary()
        self._build_sets()

    def _build_elementary(self):
        M = self.M
        gx, gw = leggauss(2 * M + 4)
        V, Vp = _modleg_value_deriv(M, gx)
        self.m_ref = (V * gw) @ V.T
        self.s_ref = (Vp * gw) @ Vp.T
        self.c_ref = (Vp * gw) @ V.T
        Ve, _ = _modleg_value_deriv(M, np.array([-1.0, 1.0]))
        self.val_m1 = Ve[:, 0]
        self.val_p1 = Ve[:, 1]

    def _build_sets(self):
        N, M, tau = self.N, self.M, self.tau
        til = []
        for node in range(N):
            S = np.zeros((N, M), dtype=_C)
            left_seg = (node - 1) % N
            right_seg = node
            if node == 0:
                S[right_seg, 0] += 1.0
                S[left_seg, 1] += tau
            else:
                S[left_seg, 1] += 1.0
                S[right_seg, 0] += 1.0
            til.append(S)
        for seg in range(N):
            for a in range(2, M):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                til.append(S)
        self.Btilde = til
        b = []
        for seg in range(N):
            for a in (0, 1):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
            for a in range(2, M - 1):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
        self.B = b
        assert len(self.Btilde) == len(self.B) == N * (M - 1)
        self.dim = N * (M - 1)


# =========================================================================== #
# NEW: general per-segment Galerkin pair + tensor-weighted 2-D assembly
# =========================================================================== #

def _seg_pair(basis, lset, op, rset):
    """Per-segment matrices ``G[s][i,j]`` of ``<lset_i | op | rset_j>``.

    ``op``: ``'m'`` mass; ``'dL'`` derivative on the TEST (left); ``'dR'``
    derivative on the TRIAL (right).  Scales copied from the shipped
    ``_global_matrix`` / ``_eps_dir``: mass ``*J``, one-derivative ``*1``."""
    Lt = np.array(getattr(basis, lset))
    Rt = np.array(getattr(basis, rset))
    if op == "m":
        ref, scale = basis.m_ref, basis.J
    elif op == "dL":
        ref, scale = basis.c_ref, 1.0
    elif op == "dR":
        ref, scale = basis.c_ref.T, 1.0
    else:
        raise ValueError(op)
    RR = np.einsum("ab,jsb->jsa", ref, Rt)
    return scale * np.einsum("isa,jsa->sij", np.conj(Lt), RR)


def _kron2(bx, by, w, xs, ys):
    """``sum_{sx,sy} w[sx,sy] * kron(Gy[sy], Gx[sx])`` (index I = jx + qx*jy,
    the shipped tensor ordering)."""
    Gx = _seg_pair(bx, *xs)
    Gy = _seg_pair(by, *ys)
    out = np.zeros((Gy.shape[1] * Gx.shape[1], Gy.shape[2] * Gx.shape[2]),
                   dtype=_C)
    for sx in range(bx.N):
        Wy = np.einsum("y,yij->ij", np.asarray(w[sx, :], dtype=_C), Gy)
        out += np.kron(Wy, Gx[sx])
    return out


class StaggeredCell:
    """All Galerkin blocks of one staggered 2-D cell for a full 3x3 tensor.

    ``eps_cell`` is ``(Nx, Ny, 3, 3)`` PUBLIC-convention permittivity, constant
    per segment-cell.  ``kx0``/``ky0`` are the incident transverse wavenumbers
    in ``k0`` units (physical sign; the Bloch glue is ``exp(+i k0 kx0 px)``)."""

    def __init__(self, px, py, eps_cell, M, k0, kx0=0.0, ky0=0.0):
        eps_cell = np.asarray(eps_cell, dtype=_C)
        assert eps_cell.ndim == 4 and eps_cell.shape[2:] == (3, 3)
        Nx, Ny = eps_cell.shape[:2]
        assert Nx == Ny, "square grid (Nx*(M-1) == Ny*(M-1))"
        self.px, self.py, self.M, self.k0 = float(px), float(py), int(M), float(k0)
        self.kx0, self.ky0 = float(kx0), float(ky0)
        self.eps = eps_cell
        self.Nx, self.Ny = Nx, Ny
        self.bx = Basis1D(px, Nx, M, np.exp(1j * k0 * kx0 * px))
        self.by = Basis1D(py, Ny, M, np.exp(1j * k0 * ky0 * py))
        self.bx._K0 = float(kx0)          # Bloch wavenumber in k0 units
        self.by._K0 = float(ky0)
        self.q = self.bx.dim
        self.qq = self.q * self.q
        self._assemble()

    def _assemble(self):
        bx, by, k0 = self.bx, self.by, self.k0
        e = self.eps
        one = np.ones((self.Nx, self.Ny), dtype=_C)
        K = lambda w, xs, ys: _kron2(bx, by, w, xs, ys)      # noqa: E731
        T, B = "Btilde", "B"
        # --- unweighted Grams -------------------------------------------
        self.M1 = K(one, (B, "m", B), (T, "m", T))           # <V1|V1>
        self.M2 = K(one, (T, "m", T), (B, "m", B))           # <V2|V2>
        self.M3 = K(one, (T, "m", T), (T, "m", T))           # <V3|V3>
        self.Mw = K(one, (B, "m", B), (B, "m", B))           # <Vw|Vw>
        # --- strong mimetic derivative operators (all /k0) ---------------
        self.CwE1 = K(one, (B, "m", B), (B, "dR", T)) / k0   # <Vw|D2|V1>
        self.CwE2 = K(one, (B, "dR", T), (B, "m", B)) / k0   # <Vw|D1|V2>
        self.P13 = K(one, (B, "dR", T), (T, "m", T)) / k0    # <V1|D1|V3>
        self.P23 = K(one, (T, "m", T), (B, "dR", T)) / k0    # <V2|D2|V3>
        # --- eps-weighted component masses (Appendix-A Eq. 40/41 + OOP) --
        ew = lambda i, j: e[:, :, i, j]                      # noqa: E731
        self.A11 = K(ew(0, 0), (B, "m", B), (T, "m", T))
        self.A12 = K(ew(0, 1), (B, "m", T), (T, "m", B))
        self.A13 = K(ew(0, 2), (B, "m", T), (T, "m", T))
        self.A21 = K(ew(1, 0), (T, "m", B), (B, "m", T))
        self.A22 = K(ew(1, 1), (T, "m", T), (B, "m", B))
        self.A23 = K(ew(1, 2), (T, "m", T), (B, "m", T))
        self.A31 = K(ew(2, 0), (T, "m", B), (T, "m", T))
        self.A32 = K(ew(2, 1), (T, "m", T), (T, "m", B))
        self.A33 = K(ew(2, 2), (T, "m", T), (T, "m", T))
        # --- eps-weighted div-D blocks, derivative on the V3 TEST (Eq. 44
        #     generalized: the third column is the NEW e13/e23 pair) -------
        self.K11 = K(ew(0, 0), (T, "dL", B), (T, "m", T)) / k0
        self.K12 = K(ew(0, 1), (T, "dL", T), (T, "m", B)) / k0
        self.K13 = K(ew(0, 2), (T, "dL", T), (T, "m", T)) / k0
        self.K21 = K(ew(1, 0), (T, "m", B), (T, "dL", T)) / k0
        self.K22 = K(ew(1, 1), (T, "m", T), (T, "dL", B)) / k0
        self.K23 = K(ew(1, 2), (T, "m", T), (T, "dL", T)) / k0
        # --- curl-curl (identical to the shipped Stt up to sign) ---------
        Mwi = np.linalg.inv(self.Mw)
        self.Sxx = self.CwE1.conj().T @ Mwi @ self.CwE1
        self.Syy = self.CwE2.conj().T @ Mwi @ self.CwE2
        self.Sxy = self.CwE1.conj().T @ Mwi @ self.CwE2
        self.Syx = self.CwE2.conj().T @ Mwi @ self.CwE1
        self._Mwi = Mwi


# =========================================================================== #
# CANDIDATE (a): first-order staggered generator on [E1; E2; G1; G2], 4 q^2
# =========================================================================== #

def generator_a(c: StaggeredCell):
    """``(Amat, Bmat)`` of the pencil ``Amat x = q Bmat x``,
    ``x = [e1; e2; g1; g2]``, ``Bmat = blkdiag(M1, M2, M2, M1)``.

    E3 eliminated WEAKLY from the longitudinal curl-H row tested in V3
    (``D1 G2 - D2 G1 = (eps E)_3``, derivative moved onto the continuous V3
    test); H3 eliminated STRONGLY (``G3 = D1 E2 - D2 E1`` lands exactly in Vw).
    """
    qq = c.qq
    Z = np.zeros((qq, qq), dtype=_C)
    # e3 = A33^-1 [ -A31, -A32, P23^H, -P13^H ] x
    E3S = np.linalg.solve(
        c.A33, np.concatenate([-c.A31, -c.A32, c.P23.conj().T,
                               -c.P13.conj().T], axis=1))
    # g3 = Mw^-1 [ -CwE1, CwE2, 0, 0 ] x
    G3S = c._Mwi @ np.concatenate([-c.CwE1, c.CwE2, Z, Z], axis=1)
    row0 = -1j * np.concatenate([Z, Z, Z, c.M1], axis=1) - 1j * (c.P13 @ E3S)
    row1 = 1j * np.concatenate([Z, Z, c.M2, Z], axis=1) - 1j * (c.P23 @ E3S)
    row2 = (-1j * np.concatenate([c.A21, c.A22, Z, Z], axis=1)
            - 1j * (c.A23 @ E3S) + 1j * (c.CwE2.conj().T @ G3S))
    row3 = (1j * np.concatenate([c.A11, c.A12, Z, Z], axis=1)
            + 1j * (c.A13 @ E3S) + 1j * (c.CwE1.conj().T @ G3S))
    Amat = np.concatenate([row0, row1, row2, row3], axis=0)
    Bmat = np.zeros((4 * qq, 4 * qq), dtype=_C)
    Bmat[0:qq, 0:qq] = c.M1
    Bmat[qq:2 * qq, qq:2 * qq] = c.M2
    Bmat[2 * qq:3 * qq, 2 * qq:3 * qq] = c.M2
    Bmat[3 * qq:, 3 * qq:] = c.M1
    return Amat, Bmat, E3S, G3S


def modes_a(c: StaggeredCell):
    """All ``4 q^2`` modes of candidate (a): ``(W, V, qv, e3)`` with
    ``W = [e1; e2]``, ``V = [g1; g2]``, ``qv = gamma/k0``."""
    Amat, Bmat, E3S, _G3S = generator_a(c)
    L = np.linalg.cholesky(Bmat)                 # Bmat HPD (block Gram)
    Ah = sla.solve_triangular(L, Amat, lower=True)
    Ah = sla.solve_triangular(L, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = sla.solve_triangular(L.conj().T, Y, lower=False)
    qq = c.qq
    W = X[:2 * qq, :]
    V = X[2 * qq:, :]
    e3 = E3S @ X
    return W, V, qv, e3


# =========================================================================== #
# CANDIDATE (d): quadratic-in-gamma E-form keeping div(D) = 0, 6 q^2 pencil
# =========================================================================== #

def pencil_d(c: StaggeredCell):
    """``(P2, P1, P0)`` of ``q^2 P2 x + q P1 x + P0 x = 0``,
    ``x = [e1; e2; e3]``."""
    qq = c.qq
    Z = np.zeros((qq, qq), dtype=_C)
    P2 = np.block([[c.M1, Z, Z], [Z, c.M2, Z], [Z, Z, Z]])
    P1 = 1j * np.block([[Z, Z, c.P13],
                        [Z, Z, c.P23],
                        [c.A31, c.A32, c.A33]])
    P0 = np.block([
        [c.Sxx - c.A11, -c.Sxy - c.A12, -c.A13],
        [-c.Syx - c.A21, c.Syy - c.A22, -c.A23],
        [-(c.K11 + c.K21), -(c.K12 + c.K22), -(c.K13 + c.K23)],
    ])
    return P2, P1, P0


def modes_d(c: StaggeredCell, *, qcut=1e8):
    """All FINITE modes of candidate (d) from the ``6 q^2`` linearization.

    Returns ``(W, V, qv, e3, n_inf)`` -- ``n_inf`` = eigenvalues rejected as
    infinite (the singular leading coefficient ``P2``)."""
    P2, P1, P0 = pencil_d(c)
    n = P0.shape[0]
    I = np.eye(n, dtype=_C)
    Z = np.zeros((n, n), dtype=_C)
    A6 = np.block([[Z, I], [-P0, -P1]])
    B6 = np.block([[I, Z], [Z, P2]])
    w, Vr = sla.eig(A6, B6, right=True, homogeneous_eigvals=True)
    alpha, beta = w[0], w[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        qv = alpha / beta
    good = np.isfinite(qv) & (np.abs(qv) < qcut)
    n_inf = int((~good).sum())
    qv = qv[good]
    X = Vr[:n, good]                       # x = [e1; e2; e3]
    qq = c.qq
    e1, e2, e3 = X[:qq], X[qq:2 * qq], X[2 * qq:]
    W = np.concatenate([e1, e2], axis=0)
    V = h_partner(c, W, e3, qv)
    return W, V, qv, e3, n_inf


# =========================================================================== #
# Shared: strong H-partner, flux, forward/backward split
# =========================================================================== #

# candidate (d) carries exactly q^2 EXTRA modes at gamma = 0 (measured
# |q| <= 4.3e-13 at M=7 vs a smallest physical |q| of ~1.4 -- 12 decades of
# gap; see m1 T2b).  They are the static null space of the div(D)=0
# constraint and must be dropped before the forward/backward split.
_NULL_TOL = 1e-6


def h_partner(c: StaggeredCell, W, e3, qv):
    """``V = [g1; g2]`` from the STRONG transverse curl-E rows
    ``G1 = D2 E3 - i q E2`` (in V2) and ``G2 = i q E1 - D1 E3`` (in V1)."""
    qq = c.qq
    e1, e2 = W[:qq], W[qq:]
    g1 = np.linalg.solve(c.M2, c.P23 @ e3 - 1j * (c.M2 @ e2) * qv[None, :])
    g2 = np.linalg.solve(c.M1, 1j * (c.M1 @ e1) * qv[None, :] - c.P13 @ e3)
    return np.concatenate([g1, g2], axis=0)


def modal_flux(c: StaggeredCell, W, V):
    """``Sz = Im(e1^H M1 g2 - e2^H M2 g1)`` -- the z-Poynting flux in the
    inner product MATCHING the basis (PMM_ROADMAP C-FLUX)."""
    qq = c.qq
    e1, e2 = W[:qq], W[qq:]
    g1, g2 = V[:qq], V[qq:]
    return np.imag(np.sum(np.conj(e1) * (c.M1 @ g2), axis=0)
                   - np.sum(np.conj(e2) * (c.M2 @ g1), axis=0))


def split_forward(c: StaggeredCell, W, V, qv):
    """Forward / backward split via the SHIPPED generalized flux selector
    (``rcwa._core._select_forward_flux``), fed the Cholesky-whitened blocks so
    its plain harmonic sums reproduce the Gram-weighted flux exactly."""
    from lumenairy.elements.rcwa._core import _select_forward_flux
    qq = c.qq
    L1 = np.linalg.cholesky(c.M1).conj().T          # M1 = L1^H L1
    L2 = np.linalg.cholesky(c.M2).conj().T
    Vfull = np.concatenate([L1 @ W[:qq], L2 @ W[qq:],
                            L2 @ V[:qq], L1 @ V[qq:]], axis=0)
    nrm = np.linalg.norm(Vfull, axis=0)
    Vfull = Vfull / np.where(nrm == 0, 1.0, nrm)[None, :]
    lam = -1j * qv                                   # exp(-lam k0 z) forward
    fidx = _select_forward_flux(lam, Vfull, qq)
    fidx = np.asarray(fidx)
    bidx = np.array(sorted(set(range(qv.size)) - set(fidx.tolist())), dtype=int)
    return fidx, bidx


# =========================================================================== #
# IN-PLANE / ISOTROPIC E-form (2 q^2) -- the shipped discretization, rebuilt
# here in the probe's sign convention (used for the half-spaces and for M3)
# =========================================================================== #

def eform_operators(c: StaggeredCell):
    """``(L, G)`` of the shipped Granet E-form ``L e = q^2 G e`` derived as the
    OOP-free reduction of candidate (d) (proved algebraically in the doc)."""
    Kdiv = np.concatenate([c.K11 + c.K21, c.K12 + c.K22], axis=1)
    Ktz = np.concatenate([c.P13, c.P23], axis=0)
    Schur = Ktz @ np.linalg.solve(c.A33, Kdiv)
    L = np.block([[c.A11 - c.Sxx, c.A12 + c.Sxy],
                  [c.A21 + c.Syx, c.A22 - c.Syy]]) - Schur
    G = np.zeros_like(L)
    qq = c.qq
    G[:qq, :qq] = c.M1
    G[qq:, qq:] = c.M2
    return L, G


def eform_modes(c: StaggeredCell):
    """Forward E-form modes ``(W, V, qv, e3)`` (2 q^2), branch-selected exactly
    as the shipped ``_region_modes``."""
    from lumenairy.elements.pmm._core import _forward_branch_flip
    L, G = eform_operators(c)
    g2v, W = sla.eig(L, G)
    qv = _forward_branch_flip(np.sqrt(np.asarray(g2v, dtype=_C)))
    Kdiv = np.concatenate([c.K11 + c.K21, c.K12 + c.K22], axis=1)
    iqe3 = np.linalg.solve(c.A33, Kdiv @ W)
    safe = np.where(np.abs(qv) < 1e-12, 1e-12, qv)
    e3 = iqe3 / (1j * safe)[None, :]
    V = h_partner(c, W, e3, qv)
    return W, V, qv, e3


# =========================================================================== #
# Exact dispersion:  det(k k^T - |k|^2 I + eps) = 0  (PUBLIC eps, k in k0)
# =========================================================================== #

def exact_kz_roots(eps, u, v):
    """Four exact ``kz/k0`` roots for transverse ``(u, v)`` (in ``k0`` units).

    The 3x3 determinant is expanded in EXACT polynomial arithmetic (each entry
    of ``k k^T - |k|^2 I + eps`` is a polynomial of degree <= 2 in ``kz``), not
    sampled-and-fitted: the sampled version used by
    ``tests/unit/test_audit_oop_dispersion.py`` conditions the Vandermonde at
    ~1e-8, which is decades above the bar this probe needs."""
    from numpy.polynomial.polynomial import polyadd, polymul, polysub
    e = np.asarray(eps, dtype=_C)
    P = np.empty((3, 3), dtype=object)
    P[0, 0] = np.array([e[0, 0] - v * v, 0.0, -1.0], dtype=_C)
    P[0, 1] = np.array([e[0, 1] + u * v], dtype=_C)
    P[0, 2] = np.array([e[0, 2], u], dtype=_C)
    P[1, 0] = np.array([e[1, 0] + u * v], dtype=_C)
    P[1, 1] = np.array([e[1, 1] - u * u, 0.0, -1.0], dtype=_C)
    P[1, 2] = np.array([e[1, 2], v], dtype=_C)
    P[2, 0] = np.array([e[2, 0], u], dtype=_C)
    P[2, 1] = np.array([e[2, 1], v], dtype=_C)
    P[2, 2] = np.array([e[2, 2] - u * u - v * v], dtype=_C)
    det = polyadd(
        polysub(polymul(P[0, 0], polysub(polymul(P[1, 1], P[2, 2]),
                                         polymul(P[1, 2], P[2, 1]))),
                polymul(P[0, 1], polysub(polymul(P[1, 0], P[2, 2]),
                                         polymul(P[1, 2], P[2, 0])))),
        polymul(P[0, 2], polysub(polymul(P[1, 0], P[2, 1]),
                                 polymul(P[1, 1], P[2, 0]))))
    return np.roots(det[::-1])


# =========================================================================== #
# Far field: Fourier -> Rayleigh projection in the PROBE's sign convention
# =========================================================================== #

def _fourier_projection(basis: Basis1D, orders, K):
    """``T[m, j] = (1/d) INT phi_j(x) exp(-i (K + m G) x) dx`` -- extracts the
    amplitude of the PHYSICAL harmonic ``exp(+i(K + m G) x)``.  ``K`` is the
    Bloch wavenumber in 1/length (``= k0 kx0``)."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    J = basis.J
    for seg in range(N):
        xphys = 0.5 * (basis.xb[seg] + basis.xb[seg + 1]) + J * xg
        phase = np.exp(-1j * np.outer(orders * G + K, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

    def asm(global_set):
        S = np.array(global_set)
        return np.einsum("msa,jsa->mj", T_local, S)
    return asm


def far_projectors(c: StaggeredCell, ox, oy):
    Kx = c.k0 * c.kx0
    Ky = c.k0 * c.ky0
    ax = _fourier_projection(c.bx, ox, Kx)
    ay = _fourier_projection(c.by, oy, Ky)
    P1 = np.kron(ay(c.by.Btilde), ax(c.bx.B))       # V1 = B(x) x Btil(y)
    P2 = np.kron(ay(c.by.B), ax(c.bx.Btilde))       # V2 = Btil(x) x B(y)
    return P1, P2


def _kz_fwd(eps, kx, ky):
    val = np.sqrt(np.asarray(eps - kx ** 2 - ky ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


# =========================================================================== #
# Single-layer Jones / R / T driver (the stack2d_pure.solve pattern)
# =========================================================================== #

def solve_slab(px, py, eps_cell, n_substrate, n_superstrate, depth, wl, *,
               M=6, n_orders=3, theta=0.0, phi=0.0, candidate="a",
               return_modes=False):
    """R/T/Jones of ONE tensor layer between isotropic half-spaces.

    Argument order MATCHES the shipped ``pmm_efficiency_2d_staggered``
    (``..., n_substrate, n_superstrate, depth, wavelength``) -- a probe that
    took ``(n_sup, n_sub)`` instead cost one bisection round.

    ``candidate`` in ``{'a', 'd', 'eform'}``.  Returns
    ``(orders (Nfo,2), R (2,Nfo), T (2,Nfo), jones (2,2))``; rows = incident
    Ex / Ey; ``jones`` = order-0 REFLECTION, PUBLIC convention."""
    from lumenairy.elements.pmm._core import _guarded_lstsq
    from lumenairy.elements.rcwa._core import (
        _interface_smatrix_general,
        _modes_to_M,
        _project_efficiency,
        _propagation_smatrix_general,
        _redheffer_star,
    )

    eps_cell = np.asarray(eps_cell, dtype=_C)
    Nx, Ny = eps_cell.shape[:2]
    eps_sup, eps_sub = _C(n_superstrate) ** 2, _C(n_substrate) ** 2
    k0 = 2.0 * np.pi / wl
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    lay = StaggeredCell(px, py, eps_cell, M, k0, kx0, ky0)
    # The once-only forward Rayleigh projection is a LEAST-SQUARES match of the
    # incident plane wave onto the region's modal basis: it must be
    # OVER-determined, i.e. (2*n_orders+1) >= q per axis.  Under-determining it
    # (the default n_orders=3 at q=12) hands lstsq a min-norm solution whose
    # null-space part is scattered by S11 -- measured as a 6e-3 R error and an
    # O(1) Jones phase on an ISOTROPIC slab at oblique (m2 bisection).
    n_orders = max(int(n_orders), int(np.ceil((lay.q - 1) / 2)))
    iso = lambda e: np.broadcast_to(                       # noqa: E731
        _C(e) * np.eye(3, dtype=_C), (Nx, Ny, 3, 3)).copy()
    sup = StaggeredCell(px, py, iso(eps_sup), M, k0, kx0, ky0)
    sub = StaggeredCell(px, py, iso(eps_sub), M, k0, kx0, ky0)

    Wsup, Vsup, lsup, _ = eform_modes(sup)
    Wsub, Vsub, lsub, _ = eform_modes(sub)
    Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
    Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)

    if candidate == "a":
        W, V, qv, _e3 = modes_a(lay)
    elif candidate == "d":
        W, V, qv, _e3, _ni = modes_d(lay)
        keep = np.abs(qv) > _NULL_TOL * max(1.0, float(np.max(np.abs(qv))))
        W, V, qv = W[:, keep], V[:, keep], qv[keep]
    elif candidate == "eform":
        W, V, qv, _e3 = eform_modes(lay)
        W = np.concatenate([W, W], axis=1)
        V = np.concatenate([V, -V], axis=1)
        qv = np.concatenate([qv, -qv])
    else:
        raise ValueError(candidate)
    fidx, bidx = split_forward(lay, W, V, qv)
    nf = 2 * lay.qq
    if fidx.size != nf or bidx.size != nf:
        raise RuntimeError(f"forward/backward split {fidx.size}/{bidx.size} "
                           f"!= {nf}/{nf} (candidate {candidate})")
    Mlay = _modes_to_M(W[:, fidx], V[:, fidx], W[:, bidx], V[:, bidx])
    lam_f = -1j * qv[fidx]
    lam_b = -1j * qv[bidx]

    S = _interface_smatrix_general(Msup, Mlay)
    S = _redheffer_star(S, _propagation_smatrix_general(lam_f, lam_b,
                                                        k0 * depth))
    S = _redheffer_star(S, _interface_smatrix_general(Mlay, Msub))
    S11, _S12, S21, _S22 = S

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = len(order_x)
    P1, P2 = far_projectors(sup, ox, oy)
    qq = lay.qq
    Hsup = np.concatenate([P1 @ Wsup[:qq, :], P2 @ Wsup[qq:, :]], axis=0)
    Hsub = np.concatenate([P1 @ Wsub[:qq, :], P2 @ Wsub[qq:, :]], axis=0)
    kxv = kx0 + order_x * (wl / px)
    kyv = ky0 + order_y * (wl / py)
    kz_ref = _kz_fwd(eps_sup, kxv, kyv)
    kz_trn = _kz_fwd(eps_sub, kxv, kyv)
    kz_inc = float(np.real(_kz_fwd(eps_sup, kx0, ky0)))
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

    R_rows, T_rows, j_cols = [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        long_inc = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        rhs = np.concatenate([ex0 * delta, ey0 * delta])
        cinc = _guarded_lstsq(Hsup, rhs, "probe far-field Rayleigh projection")
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:Nfo], r_ord[Nfo:]
        tx, ty = t_ord[:Nfo], t_ord[Nfo:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re_, Te_ = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                       rx, ry, rz, tx, ty, tz, einc_sq)
        R_rows.append(Re_)
        T_rows.append(Te_)
        j_cols.append(np.stack([rx[p0], ry[p0]]))
    out = (np.stack([order_x, order_y], axis=1), np.stack(R_rows),
           np.stack(T_rows), np.stack(j_cols, axis=1))
    if return_modes:
        return out + (dict(lay=lay, W=W, V=V, qv=qv, fidx=fidx, bidx=bidx),)
    return out


# =========================================================================== #
# small utilities
# =========================================================================== #

def uniaxial(no, ne, tilt_deg, azim_deg=0.0, loss=0.0):
    """Rotated uniaxial tensor, PUBLIC convention (``Im eps > 0`` = loss).
    Optic axis tilted ``tilt_deg`` from z toward x, then rotated ``azim_deg``
    about z."""
    no2 = no ** 2 + 1j * loss
    ne2 = ne ** 2 + 1j * loss
    t = np.deg2rad(tilt_deg)
    a = np.deg2rad(azim_deg)
    ct, st, ca, sa = np.cos(t), np.sin(t), np.cos(a), np.sin(a)
    axis = np.array([st * ca, st * sa, ct])
    return (no2 * np.eye(3, dtype=_C)
            + (ne2 - no2) * np.outer(axis, axis)).astype(_C)


def tile(t33, Nx, Ny):
    return np.broadcast_to(np.asarray(t33, dtype=_C),
                           (Nx, Ny, 3, 3)).copy()


def banner(name):
    print(f"# {name}")
    print(f"# lumenairy: {assert_worktree()}  v{lumenairy.__version__}")
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}")
