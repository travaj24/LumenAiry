"""Prototype: a NATIVE SLANT (constant x-z / y-z coordinate shear) for the PURE
staggered 2-D PMM (Granet 2023 modified-Legendre basis, first-order out-of-plane
generator).

FORMULATION (see docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md S1).
Frame anchored at the layer TOP, following the shipped 1-D / hybrid convention::

    u = x - t_x w ,   v = y - t_y w ,   w = z ,   t = (t_x, t_y) = tan(phi)

so ``A = d(x,y,z)/d(u,v,w) = [[1,0,tx],[0,1,ty],[0,0,1]]``, ``det A = 1``.

We use the COVARIANT field components in the frame (Granet 2017 Eq. 5 /
Li 1999), NOT the lab-Cartesian components with a chain-rule convection:

    E_cov = A^T E_lab  ->  (E_1, E_2, E_3cov) = (E_x, E_y, t.E_t + E_z)
    G_cov = A^T G_lab  ->  (G_1, G_2, G_3cov) = (G_x, G_y, t.G_t + G_z)
    G^3 (contravariant) = G_z

Maxwell in the frame is then EXACTLY the vertical system with

    curl'(E_cov) = mu^{lm} G_cov ,  curl'(G_cov) = eps^{lm} E_cov
    eps^{lm} = A^-1 eps_lab A^-T ,  mu^{lm} = g^{lm} = A^-1 A^-T   (sqrt g = 1)

Written out on the staggered de Rham placement (E1 in V1, E2 in V2, E3 in V3,
G1 in V2, G2 in V1, G3 in Vw), with ``D_3 -> i q``:

    (3)  D1 E2 - D2 E1 = G^3                                [Vw, STRONG]
    (2)  q E1 = -i G2 - i D1 E3 + i t_y G^3                 [V1]
    (1)  q E2 = +i G1 - i D2 E3 - i t_x G^3                 [V2]
    (6)  (eps E)_3 = D1 G2 - D2 G1                          [V3, WEAK]
    (5)  q G1 = -i (eps E)_2 - i D1 G_3cov                  [V2]
    (4)  q G2 = +i (eps E)_1 - i D2 G_3cov                  [V1]
      with  G_3cov = G^3 + t_x G1 + t_y G2 .

So relative to the SHIPPED vertical out-of-plane assembly the slant is exactly:
  (a) a POINTWISE congruence eps -> A^-1 eps A^-T on the cell tensor, and
  (b) six extra Galerkin blocks: two cross masses <V1|Vw>, <V2|Vw> (the
      ``t G^3`` terms in the E rows) and four single-derivative blocks
      <V2|D1|V2>, <V2|D1|V1>, <V1|D2|V2>, <V1|D2|V1> (the ``t G_t`` part of
      G_3cov in the G rows).

WHY COVARIANT AND NOT THE 1-D "CONVECTION" FORM.  The lab-Cartesian convection
``D_3 -> i q - t.D`` needs ``<B|d/dx|B>`` (the derivative of the DISCONTINUOUS
staggered set) on the E1 and G2 diagonal blocks, which is not a mimetic bracket
in this basis -- and, physically, the lab E_z JUMPS across a slanted wall while
V3 is C0, so the Cartesian placement is non-conformal at any t != 0.  The
covariant components have EXACTLY the vertical continuity structure in the
frame (E_1 jumps across u = const; E_2, E_3cov are continuous), so the shipped
de Rham placement is conformal for them and every bracket needed is one the
basis already supplies.

THE ROTATION GAUGE.  The shipped assembly carries ``_OOP_ROT_SIGN`` (a 180-deg
rotation about z between the basis and the eps_cell/far-field indexing).  Under
that rotation ``t -> -t`` as well as ``e13,e23,e31,e32 -> -``, so this module
applies BOTH once, up front, and then builds everything in the rotated gauge.
At ``t = 0`` this is bit-identical to the shipped ``_assemble_oop``.

INTERFACES.  The retained transverse state ``[E1; E2; G1; G2]`` is the covariant
tangential state, which EQUALS the lab-Cartesian tangential state (only the
NORMAL component is altered by a shear), and ``w = z`` so the interface planes
are common to both frames.  The match against isotropic half-spaces / vertical
layers is therefore the IDENTITY, exactly as the roadmap's C-FRAME contract
states.  The frame is anchored at the layer TOP, so the BOTTOM interface sits at
``u = x - t d``: for a layer bounded BELOW by a HOMOGENEOUS region that is a
gauge choice (a lateral translation maps a homogeneous half-space to itself),
which leaves R exact and T carrying a per-order unimodular phase
``exp(i alpha_m . t d)``.  Measured in m3.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.linalg as sla

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lumenairy  # noqa: E402

if not os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()):
    raise RuntimeError(
        f"probe_pmm2d_staggered_slant: lumenairy resolved to "
        f"{lumenairy.__file__!r}, outside the worktree {_ROOT!r}.")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    _OOP_H_GAUGE,
    _OOP_ROT_SIGN,
    Granet2DTransverseE,
    _far_projector_2d,
    _homog_geom_cache,
    _homog_region_modes,
    _modes_as_general,
    _pmm2d_order_kz,
    _pmm2d_project_orders,
    _region_modes,
    _region_modes_oop,
)
from lumenairy.elements.pmm._core import (  # noqa: E402
    _guarded_lstsq,
    _redheffer_star,
)
from lumenairy.elements.rcwa._core import (  # noqa: E402
    _interface_smatrix_general,
    _modes_to_M,
    _project_efficiency,
    _propagation_smatrix_general,
    _select_forward_flux,
)

__all__ = [
    "SlantSolver", "slant_region_modes", "cov_tensor", "solve_slant_stack",
    "assert_worktree",
]

_ROT = _OOP_ROT_SIGN


def assert_worktree():
    """Every script calls this; the import above already raises if it fails."""
    return os.path.abspath(lumenairy.__file__)


# --------------------------------------------------------------------------- #
# the pointwise covariant congruence  eps^{lm} = A^-1 eps_lab A^-T
# --------------------------------------------------------------------------- #
def cov_tensor(eps_lab, tx, ty):
    """``A^-1 eps A^-T`` with ``A = [[1,0,tx],[0,1,ty],[0,0,1]]`` applied to the
    trailing ``(3, 3)`` axes of ``eps_lab``.  ``sqrt(g) = det A = 1``."""
    e = np.asarray(eps_lab, dtype=_C)
    Ai = np.array([[1.0, 0.0, -tx], [0.0, 1.0, -ty], [0.0, 0.0, 1.0]],
                  dtype=_C)
    return np.einsum("mp,...pq,nq->...mn", Ai, e, Ai)


def _rot_gauge(eps_lab, tx, ty):
    """Apply the 180-degree basis rotation (``_OOP_ROT_SIGN``) to BOTH the
    out-of-plane tensor entries and the slant vector, then take the covariant
    congruence in that gauge.  Returns ``(eps_cov_rot, tx_rot, ty_rot)``."""
    e = np.array(eps_lab, dtype=_C, copy=True)
    e[..., 0, 2] *= _ROT
    e[..., 1, 2] *= _ROT
    e[..., 2, 0] *= _ROT
    e[..., 2, 1] *= _ROT
    txr, tyr = _ROT * float(tx), _ROT * float(ty)
    return cov_tensor(e, txr, tyr), txr, tyr


# --------------------------------------------------------------------------- #
# the slanted first-order staggered generator
# --------------------------------------------------------------------------- #
class SlantSolver(Granet2DTransverseE):
    """``Granet2DTransverseE`` with a constant x-z / y-z SHEAR.

    ``eps_cell`` is the LAB ``(Nx, Ny, 3, 3)`` tensor map on the frame grid (the
    walls are vertical in ``(u, v)``); ``slant = (t_x, t_y)`` are the tangents.
    The parent ``__init__`` is replicated (not called) so the out-of-plane
    generator is always taken -- a sheared isotropic cell IS an out-of-plane
    tensor cell in the frame, and at ``t = 0`` the assembly is bit-identical to
    the shipped :meth:`Granet2DTransverseE._assemble_oop`.
    """

    def __init__(self, px, py, Nx, Ny, M, eps_cell, slant=(0.0, 0.0),
                 alpha0x=0.0, alpha0y=0.0, k0=2.0 * np.pi):
        from lumenairy.elements.pmm.twod_staggered import Basis1D
        self.k0 = float(k0)
        self.alpha0x = float(alpha0x)
        self.alpha0y = float(alpha0y)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1D(px, Nx, M, taux)
        self.by = Basis1D(py, Ny, M, tauy)
        lab = np.asarray(eps_cell, dtype=_C)
        if lab.ndim == 2:                       # scalar map -> isotropic tensor
            lab = lab[..., None, None] * np.eye(3, dtype=_C)
        if lab.ndim != 4 or lab.shape[2:] != (3, 3):
            raise ValueError(f"SlantSolver: eps_cell shape {lab.shape}")
        self.eps_lab = lab
        self.slant_lab = (float(slant[0]), float(slant[1]))
        ecov, txr, tyr = _rot_gauge(lab, slant[0], slant[1])
        self.eps_cell = ecov                    # ALREADY in the rotated gauge
        self.tx, self.ty = txr, tyr
        self.q = self.bx.dim
        assert self.bx.dim == self.by.dim
        self.offplane = True
        self._assemble_slant()

    # ---------------------------------------------------------------- assembly
    def _assemble_slant(self):
        """The shipped ``_assemble_oop`` pencil with ``rot`` already folded into
        ``self.eps_cell`` (so no rot here) PLUS the six slant blocks."""
        bx, by = self.bx, self.by
        k0 = self.k0
        self._axis_mats()
        qq = self.q * self.q
        e = self.eps_cell
        tx, ty = self.tx, self.ty
        e11, e12, e13 = e[..., 0, 0], e[..., 0, 1], e[..., 0, 2]
        e21, e22, e23 = e[..., 1, 0], e[..., 1, 1], e[..., 1, 2]
        e31, e32 = e[..., 2, 0], e[..., 2, 1]
        e33 = e[..., 2, 2]

        Mtt_x, Mtt_y = self.Mtt_x, self.Mtt_y
        Mbb_x, Mbb_y = self.Mbb_x, self.Mbb_y
        dbt_x = bx.mixed(bx.B, bx.Btilde) / k0        # <B | D1 | til>_x
        dbt_y = by.mixed(by.B, by.Btilde) / k0
        Ggram1 = np.kron(Mtt_y, Mbb_x)                # <V1|V1>
        Ggram2 = np.kron(Mbb_y, Mtt_x)                # <V2|V2>
        Gw = np.kron(Mbb_y, Mbb_x)                    # <Vw|Vw>
        CwE1 = np.kron(dbt_y, Mbb_x)                  # <Vw| D2 |V1>
        CwE2 = np.kron(Mbb_y, dbt_x)                  # <Vw| D1 |V2>
        P13 = np.kron(Mtt_y, dbt_x)                   # <V1| D1 |V3>
        P23 = np.kron(dbt_y, Mtt_x)                   # <V2| D2 |V3>

        ew = self._eps_weighted
        xB, xT = bx.B, bx.Btilde
        yB, yT = by.B, by.Btilde
        mx, my = bx.m_ref, by.m_ref
        A11 = ew((bx, mx, xB, xB), (by, my, yT, yT), e11)
        A12 = ew((bx, mx, xB, xT), (by, my, yT, yB), e12)
        A13 = ew((bx, mx, xB, xT), (by, my, yT, yT), e13)
        A21 = ew((bx, mx, xT, xB), (by, my, yB, yT), e21)
        A22 = ew((bx, mx, xT, xT), (by, my, yB, yB), e22)
        A23 = ew((bx, mx, xT, xT), (by, my, yB, yT), e23)
        A31 = ew((bx, mx, xT, xB), (by, my, yT, yT), e31)
        A32 = ew((bx, mx, xT, xT), (by, my, yT, yB), e32)
        A33 = ew((bx, mx, xT, xT), (by, my, yT, yT), e33)

        Z = np.zeros((qq, qq), dtype=_C)
        E3S = np.linalg.solve(
            A33, np.concatenate([-A31, -A32, P23.conj().T, -P13.conj().T],
                                axis=1))
        G3S = np.linalg.solve(Gw, np.concatenate([-CwE1, CwE2, Z, Z], axis=1))

        row0 = (-1j * np.concatenate([Z, Z, Z, Ggram1], axis=1)
                - 1j * (P13 @ E3S))
        row1 = (1j * np.concatenate([Z, Z, Ggram2, Z], axis=1)
                - 1j * (P23 @ E3S))
        row2 = (-1j * np.concatenate([A21, A22, Z, Z], axis=1)
                - 1j * (A23 @ E3S) + 1j * (CwE2.conj().T @ G3S))
        row3 = (1j * np.concatenate([A11, A12, Z, Z], axis=1)
                + 1j * (A13 @ E3S) + 1j * (CwE1.conj().T @ G3S))

        # ------------------------------------------------ THE SIX SLANT BLOCKS
        if tx != 0.0 or ty != 0.0:
            Mtb_x = bx.mass(bx.Btilde, bx.B)          # <til|B>_x
            Mtb_y = by.mass(by.Btilde, by.B)
            Mbt_x = bx.mass(bx.B, bx.Btilde)          # <B|til>_x
            Mbt_y = by.mass(by.B, by.Btilde)
            # <til | d B> INCLUDING the jump deltas = -( <B|d til> )^H
            dtb_x = -dbt_x.conj().T
            dtb_y = -dbt_y.conj().T
            ctt_x = self.Ctt_x / k0                   # <til| D1 |til>_x
            ctt_y = self.Ctt_y / k0
            MwV1 = np.kron(Mtb_y, Mbb_x)              # <V1|Vw>
            MwV2 = np.kron(Mbb_y, Mtb_x)              # <V2|Vw>
            D1_22 = np.kron(Mbb_y, ctt_x)             # <V2|D1|V2>
            D1_21 = np.kron(Mbt_y, dtb_x)             # <V2|D1|V1>
            D2_11 = np.kron(ctt_y, Mbb_x)             # <V1|D2|V1>
            D2_12 = np.kron(dtb_y, Mbt_x)             # <V1|D2|V2>
            # E rows: +i t_y G^3 (row0, in V1) and -i t_x G^3 (row1, in V2)
            row0 = row0 + 1j * ty * (MwV1 @ G3S)
            row1 = row1 - 1j * tx * (MwV2 @ G3S)
            # G rows: G_3cov = G^3 + t_x G1 + t_y G2 under the D-bracket
            row2 = row2 - 1j * np.concatenate(
                [Z, Z, tx * D1_22, ty * D1_21], axis=1)
            row3 = row3 - 1j * np.concatenate(
                [Z, Z, tx * D2_12, ty * D2_11], axis=1)

        Bgen = np.zeros((4 * qq, 4 * qq), dtype=_C)
        Bgen[:qq, :qq] = Ggram1
        Bgen[qq:2 * qq, qq:2 * qq] = Ggram2
        Bgen[2 * qq:3 * qq, 2 * qq:3 * qq] = Ggram2
        Bgen[3 * qq:, 3 * qq:] = Ggram1
        self.Agen = np.concatenate([row0, row1, row2, row3], axis=0)
        self.Bgen = Bgen
        self.Rmat = None
        self.Lmat = None
        self.Et_blocks = None
        self.Et_offdiag = None
        self.Stt = None
        self.Schur = None
        self.dimtot = 4 * qq


def slant_region_modes(solver):
    """Forward AND backward modes of a SLANTED region as the generalized
    6-tuple ``(Wf, Vf, lam_f, Wb, Vb, lam_b)``.

    Verbatim :func:`_region_modes_oop` (Cholesky-whitened standard eig, the
    flux split with the deep-decay override, the ``_OOP_H_GAUGE`` constant) --
    reproduced here only so the prototype does not have to reach into the
    library's dispatch, which keys on ``solver.offplane``."""
    Amat, Bmat = solver.Agen, solver.Bgen
    qq = solver.q * solver.q
    Lc = np.linalg.cholesky(Bmat)
    Ah = sla.solve_triangular(Lc, Amat, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
    W = X[:2 * qq, :]
    Gst = X[2 * qq:, :]
    L1 = np.linalg.cholesky(Bmat[:qq, :qq]).conj().T
    L2 = np.linalg.cholesky(Bmat[qq:2 * qq, qq:2 * qq]).conj().T
    Vfull = np.concatenate([L1 @ W[:qq], L2 @ W[qq:],
                            L2 @ Gst[:qq], L1 @ Gst[qq:]], axis=0)
    nrm = np.linalg.norm(Vfull, axis=0)
    Vfull = Vfull / np.where(nrm == 0.0, 1.0, nrm)[None, :]
    lam = -1j * qv
    fidx = np.asarray(_select_forward_flux(lam, Vfull, qq))
    bidx = np.array(sorted(set(range(qv.size)) - set(fidx.tolist())), dtype=int)
    V = _OOP_H_GAUGE * Gst
    return (W[:, fidx], V[:, fidx], lam[fidx],
            W[:, bidx], V[:, bidx], lam[bidx],
            qv, fidx, bidx)


# --------------------------------------------------------------------------- #
# the driver:  PMM2DStackPure.solve with per-layer slant
# --------------------------------------------------------------------------- #
def solve_slant_stack(period_x, period_y, layers, n_superstrate, n_substrate,
                      wavelength, *, M=6, n_orders=5, theta=0.0, phi=0.0,
                      collect=None, frame_phase=True, phase_sign=-1.0):
    """One-layer-or-more pure staggered cascade with an optional per-layer
    ``slant = (t_x, t_y)``.

    ``layers`` is a list of dicts ``{"thickness": d, "cell": (Nx,Ny[,3,3]),
    "slant": (tx, ty)}``.  A layer with ``slant == (0, 0)`` and a scalar /
    in-plane cell runs the SHIPPED path (second-order pencil + symmetric modes
    lifted to the 6-tuple), so this driver reproduces
    ``PMM2DStackPure.solve`` exactly there.

    Returns ``(orders, R, T, jones_r, jones_t, info)`` with ``R``/``T`` shaped
    ``(2, Nfo)`` (row 0 = incident Ex) and the ORDER-0 reflection / transmission
    Jones matrices.
    """
    px, py = float(period_x), float(period_y)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2
    wl = float(wavelength)
    Nx, Ny = np.asarray(layers[0]["cell"]).shape[:2]
    k0 = 2.0 * np.pi / wl
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    a0x, a0y = kx0 * k0, ky0 * k0

    sol_h = Granet2DTransverseE(px, py, Nx, Ny, M,
                                np.full((Nx, Ny), eps_sup),
                                alpha0x=a0x, alpha0y=a0y, k0=k0)
    geom = _homog_geom_cache(sol_h)
    bx, by = sol_h.bx, sol_h.by
    del sol_h
    Wsup, Vsup, _ls = _homog_region_modes(geom, eps_sup)
    Wsub, Vsub, _lb = _homog_region_modes(geom, eps_sub)

    modes = []
    info = {"slanted": [], "spectra": []}
    for L in layers:
        cell = np.asarray(L["cell"], dtype=_C)
        tx, ty = L.get("slant", (0.0, 0.0))
        slanted = (tx != 0.0 or ty != 0.0)
        info["slanted"].append(slanted)
        if slanted:
            sol = SlantSolver(px, py, Nx, Ny, M, cell, slant=(tx, ty),
                              alpha0x=a0x, alpha0y=a0y, k0=k0)
            out = slant_region_modes(sol)
            six = out[:6]
            info["spectra"].append(out[6])
        else:
            sol = Granet2DTransverseE(px, py, Nx, Ny, M, cell,
                                      alpha0x=a0x, alpha0y=a0y, k0=k0)
            if sol.offplane:
                six = _region_modes_oop(sol)
            else:
                Wl, Vl, lam_l, _g2 = _region_modes(sol)
                six = _modes_as_general(Wl, Vl, lam_l)
            info["spectra"].append(None)
        if collect is not None:
            collect.append(sol)
        modes.append(six + (float(L["thickness"]),))

    nlay = len(modes)
    Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
    Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
    Mlay = [_modes_to_M(m[0], m[1], m[3], m[4]) for m in modes]
    ifc = [_interface_smatrix_general(Msup, Mlay[0])]
    for i in range(1, nlay):
        ifc.append(_interface_smatrix_general(Mlay[i - 1], Mlay[i]))
    ifc.append(_interface_smatrix_general(Mlay[-1], Msub))
    prop = [_propagation_smatrix_general(m[2], m[5], k0 * m[6]) for m in modes]
    S = ifc[0]
    for i in range(nlay):
        S = _redheffer_star(S, prop[i])
        S = _redheffer_star(S, ifc[i + 1])
    S11, _S12, S21, _S22 = S

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = len(order_x)
    P1, P2 = _far_projector_2d(bx, by, ox, oy, a0x, a0y)
    qq = (Nx * (M - 1)) * (Ny * (M - 1))
    Hsup = _pmm2d_project_orders(P1, P2, Wsup, qq)
    Hsub = _pmm2d_project_orders(P1, P2, Wsub, qq)
    kxv = kx0 + order_x * (wl / px)
    kyv = ky0 + order_y * (wl / py)
    kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
        eps_sup, eps_sub, kxv, kyv, kx0, ky0)
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

    # FRAME-ANCHOR PHASE.  The frame is anchored at the TOP of each slanted
    # layer, so the SUBSTRATE plane sits at u = x - sum_j t_j d_j.  For
    # HOMOGENEOUS bounding regions that lateral offset is a gauge (a shear maps
    # a homogeneous half-space to itself), and it shows up as ONE unimodular
    # diagonal phase per order on the TRANSMITTED amplitudes -- R is exact.
    shx = sum(float(L.get("slant", (0.0, 0.0))[0]) * float(L["thickness"])
              for L in layers)
    shy = sum(float(L.get("slant", (0.0, 0.0))[1]) * float(L["thickness"])
              for L in layers)
    tphase = np.ones(Nfo, dtype=_C)
    if frame_phase and (shx != 0.0 or shy != 0.0):
        tphase = np.exp(phase_sign * 1j * k0 * (kxv * shx + kyv * shy))
    info["frame_shift"] = (shx, shy)

    R_rows, T_rows, jr, jt = [], [], [], []
    amp = {k: np.zeros((2, Nfo), dtype=_C) for k in ("rx", "ry", "tx", "ty")}
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        long_inc = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        rhs = np.concatenate([ex0 * delta, ey0 * delta])
        cinc = _guarded_lstsq(Hsup, rhs, "slant probe far-field projection")
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:Nfo], r_ord[Nfo:]
        tx_, ty_ = t_ord[:Nfo] * tphase, t_ord[Nfo:] * tphase
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx_ + kyv * ty_) / safe_t
        Re, Te = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                     rx, ry, rz, tx_, ty_, tz, einc_sq)
        R_rows.append(Re)
        T_rows.append(Te)
        jr.append(np.stack([rx[p0], ry[p0]]))
        jt.append(np.stack([tx_[p0], ty_[p0]]))
        amp["rx"][col], amp["ry"][col] = rx, ry
        amp["tx"][col], amp["ty"][col] = tx_, ty_
    info["amp"] = amp
    info["kxv"], info["kyv"] = kxv, kyv
    info["dof"] = 4 * qq
    return (np.stack([order_x, order_y], axis=1),
            np.stack(R_rows), np.stack(T_rows),
            np.stack(jr, axis=1), np.stack(jt, axis=1), info)


# --------------------------------------------------------------------------- #
def tensor_uniaxial(no, ne, tilt, azim):
    """Rotated uniaxial tensor (PUBLIC convention), tilt from z, azimuth from x."""
    d = np.diag([no ** 2, no ** 2, ne ** 2]).astype(_C)
    ct, st = np.cos(tilt), np.sin(tilt)
    ca, sa = np.cos(azim), np.sin(azim)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=_C)
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], dtype=_C)
    Rm = Rz @ Ry
    return Rm @ d @ Rm.T
