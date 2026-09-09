"""PROTOTYPE -- per-layer element grids + L2 mortar for the PURE staggered
2-D PMM (``PMM2DStackPure``).  RESEARCH CODE, not library code.

Design (see docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md S2):

* Every layer carries its OWN square segmentation ``N_i`` (Granet's uniform
  lattice; ``Nx == Ny`` is a hard constraint of the staggered tensor basis) and
  its own modal count ``M_i``.
* Tangential continuity at a non-conforming interface is imposed WEAKLY:
  tangential E tested against the LOWER-index-side... no -- against grid B's
  trace space, tangential H against grid A's, exactly the pairing the shipped
  1-D ``_interface_smatrix_mortar`` uses.
* The 1-D mortar's ``kron(I_2, M)`` block operator does NOT carry over: in 2-D
  the two transverse components live in DIFFERENT tensor-product spaces
  (``V1 = B(x) (x) Btilde(y)`` for E1, ``V2 = Btilde(x) (x) B(y)`` for E2), and
  the Eq.-25 H partner SWAPS them (H1 in V2, H2 in V1).  The block operator is
  therefore ``blkdiag(Mass_V1, Mass_V2)`` on the E row and
  ``blkdiag(Mass_V2, Mass_V1)`` on the H row.
* Each 2-D cross-mass factors EXACTLY as a Kronecker product of two 1-D
  cross-masses between modified-Legendre sets on different uniform partitions,
  integrated by Gauss-Legendre on the UNION of the two partitions per axis
  (exact: the integrand is a polynomial of degree <= M_a + M_b - 2 on every
  union sub-interval).
* Half-spaces are built on the grid of the layer they touch, so both end
  interfaces are PLAIN (square) matches and the once-only far-field Rayleigh
  projection never sees a mortar.
"""
from __future__ import annotations

import os
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

WORKTREE = os.path.normcase(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def guard():
    """Every probe script calls this: refuse to run against a lumenairy that
    is not this worktree's."""
    import lumenairy
    p = os.path.normcase(os.path.abspath(lumenairy.__file__))
    if not p.startswith(WORKTREE):
        raise SystemExit(f"REFUSED: lumenairy is {p}, not under {WORKTREE}")
    return p


guard()

from lumenairy.elements.pmm._core import (  # noqa: E402
    _guarded_lstsq,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star_rect,
)
from lumenairy.elements.rcwa._core import (  # noqa: E402
    _interface_smatrix_general,
    _modes_to_M,
    _project_efficiency,
    _propagation_smatrix_general,
    _rcond_1_equilibrated,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Basis1D,
    Granet2DTransverseE,
    _far_projector_2d,
    _homog_geom_cache,
    _homog_region_modes,
    _modes_as_general,
    _modleg_value_deriv,
    _pmm2d_order_kz,
    _pmm2d_project_orders,
    _region_modes,
    _region_modes_oop,
    _tile_needs_oop,
)

__all__ = ["guard", "cross_mass_1d", "GridOps", "CrossOps",
           "interface_mortar_2d", "interface_general_mortar_2d",
           "MortarStack2D", "refine_cell", "CENSUS"]

#: M7 conditioning census.  Armed by setting to a list; every mortar solve
#: appends ``(site, n, rcond_equilibrated)``.
CENSUS = None

#: FAIL-BEFORE switch for the 2-D-specific design choice under test: the
#: Eq.-25 dual places H2 in the V1 slot and H1 in the V2 slot, so the H row of
#: the mortar must be tested with the V1/V2 block operators SWAPPED relative to
#: the E row.  ``False`` uses the naive (1-D-looking) same-order blocks -- the
#: control that shows the swap is load-bearing rather than cosmetic.
H_BLOCK_SWAP = True


# =========================================================================== #
# 1-D cross-mass between two modified-Legendre staggered sets on DIFFERENT
# uniform partitions of the same period.
# =========================================================================== #
def cross_mass_1d(ba: Basis1D, bb: Basis1D, which: str):
    """``C[i, j] = INT_0^d conj(phi^a_i(x)) phi^b_j(x) dx`` between the SAME
    named global set (``'B'`` or ``'Btilde'``) on two :class:`Basis1D`
    objects covering the same period.

    Exact by Gauss-Legendre on the UNION of the two segment partitions: on
    every union sub-interval both sides are polynomials (degree <= M-1), so
    ``M_a + M_b + 2`` points integrate the product exactly.  Near-coincident
    walls appear only in this INTEGRATION mesh -- never as spectral elements
    (the decisive property the 1-D ``_sem_cross_mass`` docstring records).

    The LEFT set is conjugated: ``Basis1D`` glues its periodic hat with
    ``tau = exp(-i alpha0 d)``, so BOTH the mass and the cross-mass are
    complex Hermitian-consistent -- unlike the 1-D nodal SEM, whose basis is
    real.  Reduces to ``ba.mass(set, set)`` bit-for-bit-in-quadrature when
    ``ba is bb``.
    """
    if abs(ba.d - bb.d) > 1e-13 * max(ba.d, 1.0):
        raise ValueError("cross_mass_1d: the two bases must span one period")
    Sa = np.asarray(getattr(ba, which))          # (dimA, Na, Ma)
    Sb = np.asarray(getattr(bb, which))          # (dimB, Nb, Mb)
    Ma, Mb = ba.M, bb.M
    nq = Ma + Mb + 2
    xg, wg = leggauss(nq)
    cuts = np.unique(np.concatenate([ba.xb, bb.xb]))
    tol = 1e-12 * ba.d
    merged = [float(cuts[0])]
    for x in cuts[1:]:
        if float(x) - merged[-1] > tol:
            merged.append(float(x))
    C = np.zeros((Sa.shape[0], Sb.shape[0]), dtype=_C)
    for u0, u1 in zip(merged[:-1], merged[1:]):
        mid = 0.5 * (u0 + u1)
        ea = min(max(int(np.searchsorted(ba.xb, mid, side="right") - 1), 0),
                 ba.N - 1)
        eb = min(max(int(np.searchsorted(bb.xb, mid, side="right") - 1), 0),
                 bb.N - 1)
        axl, axr = ba.xb[ea], ba.xb[ea + 1]
        bxl, bxr = bb.xb[eb], bb.xb[eb + 1]
        J = 0.5 * (u1 - u0)
        xphys = mid + J * xg
        Va, _ = _modleg_value_deriv(Ma, (2.0 * xphys - (axl + axr)) / (axr - axl))
        Vb, _ = _modleg_value_deriv(Mb, (2.0 * xphys - (bxl + bxr)) / (bxr - bxl))
        Ce = (Va * (wg * J)) @ Vb.T              # (Ma, Mb)
        C += np.conj(Sa[:, ea, :]) @ Ce @ Sb[:, eb, :].T
    return C


# =========================================================================== #
# Separable (Kronecker) application: the 2-D operators are NEVER materialised.
# Index convention of the eigensolver: I = jx + qx*jy, i.e. np.kron(Ky, Kx).
# =========================================================================== #
def kron_apply(Ky, Kx, X):
    """``kron(Ky, Kx) @ X`` without materialising the Kronecker product.

    ``X`` has ``qyB * qxB`` rows (``qxB = Kx.shape[1]``, ``qyB = Ky.shape[1]``);
    the result has ``qyA * qxA`` rows.  This is an IDENTITY, not an
    approximation: both 2-D spaces are tensor products and both partitions are
    rectangular, so every 2-D mass / cross-mass factors exactly.  Cost
    ``qyA qxB qyB n + qyA qxA qyB n`` against the dense form's
    ``(qyA qxA)(qyB qxB) n``, and the dense operator (``(N(M-1))^4`` entries)
    is never allocated -- 9.8 MB per component already at ``N = 4, M = 8``.
    """
    qxA, qxB = Kx.shape
    qyA, qyB = Ky.shape
    n = X.shape[1] if X.ndim == 2 else 1
    Xr = X.reshape(qyB, qxB, n)
    T = np.einsum("bx,yxn->ybn", Kx, Xr, optimize=True)      # (qyB, qxA, n)
    Y = np.einsum("ay,ybn->abn", Ky, T, optimize=True)       # (qyA, qxA, n)
    return Y.reshape(qyA * qxA, n)


class GridOps:
    """Per-grid geometry: the two 1-D bases and the FACTORS of the V1 / V2
    block Grams (``G1 = kron(Mtt_y, Mbb_x)``, ``G2 = kron(Mbb_y, Mtt_x)`` --
    exactly the ``-Rmat`` blocks the eigensolver builds)."""

    def __init__(self, period_x, period_y, N, M, taux, tauy):
        self.N, self.M = int(N), int(M)
        self.bx = Basis1D(period_x, N, M, taux)
        self.by = Basis1D(period_y, N, M, tauy)
        self.q = self.bx.dim
        self.qq = self.q * self.q
        self.Mtt_x = self.bx.mass(self.bx.Btilde, self.bx.Btilde)
        self.Mbb_x = self.bx.mass(self.bx.B, self.bx.B)
        self.Mtt_y = self.by.mass(self.by.Btilde, self.by.Btilde)
        self.Mbb_y = self.by.mass(self.by.B, self.by.B)
        self.V1 = (self.Mtt_y, self.Mbb_x)      # (Ky, Kx)
        self.V2 = (self.Mbb_y, self.Mtt_x)

    def key(self):
        return (self.N, self.M)


class CrossOps:
    """Cross-mass FACTORS between two grids: ``C1`` for the V1 (E1 / H2) space
    and ``C2`` for the V2 (E2 / H1) space, each as a ``(Ky, Kx)`` pair."""

    def __init__(self, ga: GridOps, gb: GridOps):
        Cbb_x = cross_mass_1d(ga.bx, gb.bx, "B")
        Ctt_x = cross_mass_1d(ga.bx, gb.bx, "Btilde")
        Cbb_y = cross_mass_1d(ga.by, gb.by, "B")
        Ctt_y = cross_mass_1d(ga.by, gb.by, "Btilde")
        self.C1 = (Ctt_y, Cbb_x)                # V1 = B(x) (x) Btilde(y)
        self.C2 = (Cbb_y, Ctt_x)                # V2 = Btilde(x) (x) B(y)

    @staticmethod
    def _H(pair):
        return (pair[0].conj().T, pair[1].conj().T)

    def C1H(self):
        return self._H(self.C1)

    def C2H(self):
        return self._H(self.C2)


def _blk2(op_top, op_bot, X, qq_in):
    """Apply ``blkdiag(op_top, op_bot)`` to a 2-block stacked matrix."""
    return np.concatenate((kron_apply(op_top[0], op_top[1], X[:qq_in]),
                           kron_apply(op_bot[0], op_bot[1], X[qq_in:])), axis=0)


def _census(site, A):
    if CENSUS is None:
        return
    A = np.asarray(A)
    try:
        rc = float(_rcond_1_equilibrated(A, np.linalg.inv(A)))
    except Exception:
        rc = float("nan")
    CENSUS.append((site, int(A.shape[0]), rc))


# =========================================================================== #
# The mortar interface S-matrices
# =========================================================================== #
def interface_mortar_2d(Wa, Va, Wb, Vb, ga: GridOps, gb: GridOps,
                        cr: CrossOps):
    """SQUARE (in-plane / symmetric ``+/-q``) interface between mode sets on
    DIFFERENT staggered grids.

    Weak continuity, with ``<u, v> = INT conj(u) v``::

        MassE_B  W_B (cb+ + cb-) = CrossE^H W_A (ca+ + ca-)   [2 qq_B eqs]
        MassH_A  V_A (ca+ - ca-) = CrossH   V_B (cb+ - cb-)   [2 qq_A eqs]

        MassE_X = blkdiag(G1_X, G2_X)      CrossE = blkdiag(C1, C2)
        MassH_X = blkdiag(G2_X, G1_X)      CrossH = blkdiag(C2, C1)

    -- the block SWAP on the H row is the 2-D content of the 1-D
    ``kron(I_2, .)``: the Eq.-25 dual puts H2 in the E1 (V1) placement and H1
    in the E2 (V2) placement, so testing H against grid A's traces means
    testing its first block in A's V2 space and its second in A's V1 space.
    Getting this wrong is silent at normal incidence on square-symmetric cells
    and O(1) otherwise.

    Then, exactly as in 1-D, with ``A = (MassE_B W_B)^-1 CrossE^H W_A`` and
    ``B = (MassH_A V_A)^-1 CrossH V_B``::

        S11 = (I+BA)^-1 (I-BA)   S12 = 2 (I+BA)^-1 B
        S21 = A (I + S11)        S22 = A S12 - I
    """
    lhsE = _blk2(gb.V1, gb.V2, Wb, gb.qq)
    rhsE = _blk2(cr.C1H(), cr.C2H(), Wa, ga.qq)
    _census("mortar E-row (MassE_B W_B)", lhsE)
    A_op = np.linalg.solve(lhsE, rhsE)
    hb_a = (ga.V2, ga.V1) if H_BLOCK_SWAP else (ga.V1, ga.V2)
    hb_c = (cr.C2, cr.C1) if H_BLOCK_SWAP else (cr.C1, cr.C2)
    lhsH = _blk2(hb_a[0], hb_a[1], Va, ga.qq)
    rhsH = _blk2(hb_c[0], hb_c[1], Vb, gb.qq)
    _census("mortar H-row (MassH_A V_A)", lhsH)
    B_op = np.linalg.solve(lhsH, rhsH)
    BA = B_op @ A_op
    I_a = np.eye(BA.shape[0], dtype=_C)
    _census("mortar interface (I + BA)", I_a + BA)
    X = np.linalg.solve(I_a + BA, np.concatenate((I_a - BA, B_op), axis=1))
    nc = I_a.shape[1]
    S11 = X[:, :nc].copy()
    S12 = 2.0 * X[:, nc:]
    S21 = A_op @ (I_a + S11)
    S22 = A_op @ S12 - np.eye(A_op.shape[0], dtype=_C)
    return (S11, S12, S21, S22)


def interface_general_mortar_2d(six_a, six_b, ga: GridOps, gb: GridOps,
                                cr: CrossOps):
    """GENERAL (explicit forward/backward) mortar interface -- the twin the
    out-of-plane / generalized cascade needs, mirroring the 1-D
    ``_interface_smatrix_general_mortar``::

        [ CrossE^H Wb_a  -MassE_B Wf_b ] [ca-]   [ -CrossE^H Wf_a  MassE_B Wb_b ] [ca+]
        [ MassH_A  Vb_a  -CrossH  Vf_b ] [cb+] = [ -MassH_A  Vf_a  CrossH  Vb_b ] [cb-]

    Rows ``2 qq_B + 2 qq_A``; unknowns ``ma + mb``; square whenever each region
    carries ``ma = 2 qq_A`` forward modes (true for both the in-plane pencil
    and the ``4 q^2`` out-of-plane generator after the flux split).
    """
    Wf_a, Vf_a, _lf_a, Wb_a, Vb_a, _lb_a = six_a[:6]
    Wf_b, Vf_b, _lf_b, Wb_b, Vb_b, _lb_b = six_b[:6]
    ma = Wf_a.shape[1]
    E1 = _blk2(cr.C1H(), cr.C2H(), Wb_a, ga.qq)
    E2 = -_blk2(gb.V1, gb.V2, Wf_b, gb.qq)
    E3 = -_blk2(cr.C1H(), cr.C2H(), Wf_a, ga.qq)
    E4 = _blk2(gb.V1, gb.V2, Wb_b, gb.qq)
    hb_a = (ga.V2, ga.V1) if H_BLOCK_SWAP else (ga.V1, ga.V2)
    hb_c = (cr.C2, cr.C1) if H_BLOCK_SWAP else (cr.C1, cr.C2)
    H1 = _blk2(hb_a[0], hb_a[1], Vb_a, ga.qq)
    H2 = -_blk2(hb_c[0], hb_c[1], Vf_b, gb.qq)
    H3 = -_blk2(hb_a[0], hb_a[1], Vf_a, ga.qq)
    H4 = _blk2(hb_c[0], hb_c[1], Vb_b, gb.qq)
    A = np.block([[E1, E2], [H1, H2]])
    B = np.block([[E3, E4], [H3, H4]])
    _census("general mortar interface (block system)", A)
    X = np.linalg.solve(A, B)
    return (X[:ma, :ma], X[:ma, ma:], X[ma:, :ma], X[ma:, ma:])


# =========================================================================== #
# The per-layer-grid cascade driver (the PMM2DStackPure surface, per-layer)
# =========================================================================== #
def refine_cell(cell, factor):
    """Re-express an ``(N, N)`` (or ``(N, N, 3, 3)``) cell on the ``factor``x
    refined uniform lattice -- exact, because the refined lattice contains
    every wall of the coarse one."""
    c = np.asarray(cell)
    return np.ascontiguousarray(np.repeat(np.repeat(c, factor, axis=0),
                                          factor, axis=1))


class MortarStack2D:
    """Per-layer-grid twin of :class:`PMM2DStackPure`.

    ``add_layer(thickness, eps=... | eps_cell=..., grid=N)``: a patterned layer
    takes ``N`` from its ``eps_cell``; a uniform layer takes the explicit
    ``grid`` (default: the previous layer's, else 1 -- a uniform region needs
    no walls at all, which is where the per-layer route's cheapest win lives).
    ``n_modes`` is the stack default ``M``; ``add_layer(..., n_modes=)``
    overrides it per layer.
    """

    def __init__(self, period_x, period_y=None, *, n_superstrate=1.0,
                 n_substrate=1.0, n_modes=8, n_orders=7):
        self.period_x = float(period_x)
        self.period_y = float(period_x if period_y is None else period_y)
        self.n_sup = complex(n_superstrate)
        self.n_sub = complex(n_substrate)
        self.M = int(n_modes)
        self.n_orders = int(n_orders)
        self._layers = []
        self._src = None
        self.timing = {}

    def add_layer(self, thickness, *, eps=None, eps_cell=None, grid=None,
                  n_modes=None):
        if (eps is None) == (eps_cell is None):
            raise ValueError("pass exactly one of eps / eps_cell")
        M = int(self.M if n_modes is None else n_modes)
        if eps_cell is not None:
            cell = np.asarray(eps_cell, dtype=_C)
            if cell.shape[0] != cell.shape[1]:
                raise ValueError("eps_cell must be square (Nx == Ny)")
            N = cell.shape[0]
            self._layers.append(dict(kind="patterned", thickness=float(thickness),
                                     eps_cell=cell, N=N, M=M))
        else:
            e = np.asarray(eps, dtype=_C)
            if grid is None:
                grid = self._layers[-1]["N"] if self._layers else 1
            N = int(grid)
            if e.ndim == 0:
                self._layers.append(dict(kind="uniform", thickness=float(thickness),
                                         eps=_C(eps), N=N, M=M))
            else:
                self._layers.append(dict(kind="uniform_tensor",
                                         thickness=float(thickness),
                                         eps33=e, N=N, M=M))
        return self

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        self._src = dict(wl=float(wavelength), theta=float(theta),
                         phi=float(phi))
        return self

    # ---------------------------------------------------------------- solve
    def solve(self, *, jones=True, force_general=False,
              force_mortar=False):
        t_all = time.perf_counter()
        wl = self._src["wl"]
        theta, phi = self._src["theta"], self._src["phi"]
        px, py = self.period_x, self.period_y
        eps_sup = _C(self.n_sup) ** 2
        eps_sub = _C(self.n_sub) ** 2
        k0 = 2.0 * np.pi / wl
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        a0x, a0y = kx0 * k0, ky0 * k0
        taux = np.exp(-1j * a0x * px)
        tauy = np.exp(-1j * a0y * py)

        # ---- per-grid geometry (bases + Gram factors), one per (N, M) ------
        t0 = time.perf_counter()
        grids = {}

        def _grid(N, M):
            g = grids.get((N, M))
            if g is None:
                g = GridOps(px, py, N, M, taux, tauy)
                grids[(N, M)] = g
            return g

        gof = [_grid(L["N"], L["M"]) for L in self._layers]
        g_sup, g_sub = gof[0], gof[-1]
        self.timing["geometry"] = time.perf_counter() - t0

        # ---- per-grid eps-free geometric eig (half-spaces + uniform scalar)
        t0 = time.perf_counter()
        geo_cache = {}

        def _geo(g):
            hit = geo_cache.get(g.key())
            if hit is None:
                # assembled with eps_sup exactly as PMM2DStackPure does, so
                # a CONFORMING per-layer stack reproduces the shipped
                # arithmetic (the split is eps-free, but not bitwise so)
                sol = Granet2DTransverseE(px, py, g.N, g.N, g.M,
                                          np.full((g.N, g.N), eps_sup),
                                          alpha0x=a0x, alpha0y=a0y, k0=k0)
                hit = _homog_geom_cache(sol)
                geo_cache[g.key()] = hit
            return hit

        Wsup, Vsup, lam_sup = _homog_region_modes(_geo(g_sup), eps_sup)
        Wsub, Vsub, lam_sub = _homog_region_modes(_geo(g_sub), eps_sub)
        self.timing["halfspace_eig"] = time.perf_counter() - t0

        # ---- per-layer modes ----------------------------------------------
        t0 = time.perf_counter()
        modes, any_oop, eig_cache = [], False, {}
        for L, g in zip(self._layers, gof):
            if L["kind"] == "uniform":
                W, V, lam = _homog_region_modes(_geo(g), L["eps"])
                six = _modes_as_general(W, V, lam)
            else:
                if L["kind"] == "uniform_tensor":
                    cell = np.ascontiguousarray(
                        np.broadcast_to(L["eps33"], (g.N, g.N, 3, 3)))
                else:
                    cell = L["eps_cell"]
                key = (g.key(), cell.shape, cell.tobytes())
                cached = eig_cache.get(key)
                if cached is None:
                    sol = Granet2DTransverseE(px, py, g.N, g.N, g.M, cell,
                                              alpha0x=a0x, alpha0y=a0y, k0=k0)
                    if sol.offplane:
                        cached = _region_modes_oop(sol)
                    else:
                        Wl, Vl, lam_l, _g2 = _region_modes(sol)
                        cached = _modes_as_general(Wl, Vl, lam_l)
                    eig_cache[key] = cached
                six = cached
                if cell.ndim == 4 and _tile_needs_oop("MortarStack2D", cell):
                    any_oop = True
            modes.append(six + (L["thickness"],))
        self.timing["layer_eig"] = time.perf_counter() - t0
        general = any_oop or force_general

        # ---- interfaces: plain when the two grids are identical -----------
        t0 = time.perf_counter()
        cross_cache = {}

        def _cross(ga, gb):
            key = (ga.key(), gb.key())
            hit = cross_cache.get(key)
            if hit is None:
                hit = CrossOps(ga, gb)
                cross_cache[key] = hit
            return hit

        nlay = len(modes)

        def _ifc(ia, ib):
            """Interface between region ia and region ib; ``None`` = half-space
            (built on the adjacent layer's grid, hence always conforming)."""
            if ia is None:
                ga, gb = g_sup, gof[0]
                sa = (Wsup, Vsup, lam_sup, Wsup, -Vsup, -lam_sup)
                sb = modes[0][:6]
            elif ib is None:
                ga, gb = gof[-1], g_sub
                sa = modes[-1][:6]
                sb = (Wsub, Vsub, lam_sub, Wsub, -Vsub, -lam_sub)
            else:
                ga, gb = gof[ia], gof[ib]
                sa, sb = modes[ia][:6], modes[ib][:6]
            same = (ga.key() == gb.key()) and not force_mortar
            if general:
                if same:
                    return _interface_smatrix_general(
                        _modes_to_M(sa[0], sa[1], sa[3], sa[4]),
                        _modes_to_M(sb[0], sb[1], sb[3], sb[4]))
                return interface_general_mortar_2d(sa, sb, ga, gb,
                                                   _cross(ga, gb))
            if same:
                return _interface_smatrix(sa[0], sa[1], sb[0], sb[1])
            return interface_mortar_2d(sa[0], sa[1], sb[0], sb[1], ga, gb,
                                       _cross(ga, gb))

        ifc = [_ifc(None, 0)]
        for i in range(1, nlay):
            ifc.append(_ifc(i - 1, i))
        ifc.append(_ifc(nlay - 1, None))
        if general:
            prop = [_propagation_smatrix_general(m[2], m[5], k0 * m[6])
                    for m in modes]
        else:
            prop = [_propagation_smatrix(m[2], k0 * m[6]) for m in modes]
        self.timing["interfaces"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        S = ifc[0]
        for i in range(nlay):
            S = _redheffer_star_rect(S, prop[i])
            S = _redheffer_star_rect(S, ifc[i + 1])
        S11, _S12, S21, _S22 = S
        self.timing["cascade"] = time.perf_counter() - t0

        # ---- far field, once, on the two END grids ------------------------
        # T3-3 lesson: the Rayleigh-order cap must be derived from the grids
        # the half-spaces actually live on, never from a union.
        cap = min((g_sup.q - 1) // 2, (g_sub.q - 1) // 2)
        n_orders = min(self.n_orders, cap)
        self.n_orders_used = n_orders
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nfo = len(order_x)
        P1s, P2s = _far_projector_2d(g_sup.bx, g_sup.by, ox, oy, a0x, a0y)
        P1t, P2t = _far_projector_2d(g_sub.bx, g_sub.by, ox, oy, a0x, a0y)
        Hsup = _pmm2d_project_orders(P1s, P2s, Wsup, g_sup.qq)
        Hsub = _pmm2d_project_orders(P1t, P2t, Wsub, g_sub.qq)
        kxv = kx0 + order_x * (wl / px)
        kyv = ky0 + order_y * (wl / py)
        kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
            eps_sup, eps_sub, kxv, kyv, kx0, ky0)
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
        R_rows, T_rows, j_cols = [], [], []
        for (ex0, ey0) in ((1.0, 0.0), (0.0, 1.0)):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            rhs = np.concatenate([ex0 * delta, ey0 * delta])
            cinc = _guarded_lstsq(Hsup, rhs, "MortarStack2D far-field")
            r_ord = Hsup @ (S11 @ cinc)
            t_ord = Hsub @ (S21 @ cinc)
            rx, ry = r_ord[:Nfo], r_ord[Nfo:]
            tx, ty = t_ord[:Nfo], t_ord[Nfo:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re, Te = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                         rx, ry, rz, tx, ty, tz, einc_sq)
            R_rows.append(Re)
            T_rows.append(Te)
            j_cols.append(np.stack([rx[p0], ry[p0]]))
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jmat = np.stack(j_cols, axis=1)
        orders2d = np.stack([order_x, order_y], axis=1)
        self.timing["total"] = time.perf_counter() - t_all
        if not jones:
            return orders2d, R_eff, T_eff
        return orders2d, R_eff, T_eff, jmat
