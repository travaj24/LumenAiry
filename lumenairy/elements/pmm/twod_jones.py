"""
lumenairy.elements.pmm.twod_jones -- anisotropic 2-D hybrid PMM (Jones).
========================================================================

:func:`pmm_jones_2d` is the full-tensor counterpart of
:func:`lumenairy.elements.pmm.pmm_efficiency_2d_cell` and the PMM mirror of
:func:`lumenairy.elements.rcwa.rcwa_jones_2d`: a single doubly-periodic layer
whose permittivity is an IN-PLANE (3, 3) tensor field over an axis-aligned
piecewise-constant cell, driven by both incident linear polarizations, returning
per-order efficiencies plus the 2x2 zeroth-order Jones reflection matrix.

Method
------
The geometry pipeline is the hybrid PMM's (exact spectral-element walls from the
pixel grid, Fourier-Galerkin projection into the Rayleigh basis); the modal
eigenproblem is the SHARED dimension-agnostic tensor block solve
(:func:`lumenairy.elements.rcwa._core._layer_eigenmodes_tensor`) -- the same
``Q``/``P`` structure :func:`rcwa_jones_2d` uses (Li 2003 z-decoupled subset),
fed with the PMM's projected nodal operators instead of Fourier convolutions.

Factorization (Li 1997, JOSA A 14:2758)
---------------------------------------
The tensor ``Q`` block applies the DIRECT (Laurent) rule to every tensor
component -- exactly :func:`rcwa_jones_2d`'s choice, exactly energy-conserving
for a lossless tensor, and reducing a scalar cell EXACTLY to
``pmm_efficiency_2d_cell(formulation='laurent')``.  The ``E_z`` elimination rule
is selectable: ``formulation='laurent'`` uses ``inv([[e_zz]])`` (the
rcwa_jones_2d mirror); ``formulation='li'`` uses the projected multiply-by-
``1/e_zz`` directly (the hybrid scalar path's validated inverse-rule
elimination).  NB Li 1997 Eqs. (8)/(9) + (31) show the OPTIMAL crossed-grating
rule gives each diagonal slot the inverse rule along its own axis and Laurent
along the other (the mixed composites); that per-direction refinement is NOT
implemented -- patterned tensor cells converge at the Laurent (~1e-3) floor.

Scope
-----
FULL (3, 3) tensors, in-plane OR out-of-plane.  An out-of-plane cell
(``xz/yz/zx/zy`` nonzero) routes through the shared full-3x3 FIRST-ORDER
GENERATOR (Li 2003; ``rcwa._core._layer_eigenmodes_tensor``'s 6-tuple branch)
and the GENERALIZED S-matrix -- forward and backward modes are genuinely
distinct there (the ``[W; -V] <-> -lam`` symmetry is broken).  This is the
library's first 2-D out-of-plane solver (``rcwa_jones_2d`` is in-plane only),
so its validation chain is 1-D-reducible cells + the Berreman-grade uniform
limit.  COST: the out-of-plane eig is ``4*Nf`` (vs ``2*Nf`` in-plane) --
~8x slower; ~14 s/layer at ``n_orders=11`` -- prefer modest ``n_orders``.
NON-RECIPROCAL cells (``e_xz != e_zx`` asymmetric, non-Hermitian) can give
``R+T != 1`` PHYSICALLY (no auto-balance) -- match against a 1-D oracle, do
not assert unity.  Loss convention: PUBLIC ``Im(eps) > 0`` for loss (the
conjugation bridge is internal, matching the rest of the suite).
"""
from __future__ import annotations

import numpy as np

from ..rcwa._core import (
    _grazing_safe_wavelength,
    _interface_smatrix_general,
    _layer_eigenmodes_tensor,
    _modes_to_M,
    _propagation_smatrix_general,
    _require_propagating_incidence,
)
from ._core import (
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
    _stabilize_jones,
)
from .twod import (
    _C,
    _MAX_NODAL_DOF,
    _PASSIVE_TOL_2D,
    _PER_ORDER_TOL_2D,
    _axis_elem_counts,
    _axis_projection,
    _build_axis,
    _cell_to_walls_tile,
    _homogeneous_modes,
    _kz_forward2,
    _scan_solver,
    _validate_cell_cost,
    _validate_cell_orders,
)

__all__ = ["pmm_jones_2d"]


def _require_inplane_tile(fn_name, tile33):
    """In-plane (z-decoupled) tensor guard for callers that do NOT support the
    generator path (the stack): xz/yz/zx/zy must vanish.  ``pmm_jones_2d``
    itself supports full out-of-plane tensors and uses only
    :func:`_require_nonzero_ezz`."""
    off = np.abs(tile33[..., [0, 1, 2, 2], [2, 2, 0, 1]])
    if float(np.max(off)) > 0.0:
        raise NotImplementedError(
            f"{fn_name}: out-of-plane tensor entries (xz/yz/zx/zy) are not "
            f"supported here -- in-plane (xx, xy, yx, yy, zz) only.  For a "
            f"SINGLE out-of-plane layer use pmm_jones_2d (the full-3x3 "
            f"generator path).")
    _require_nonzero_ezz(fn_name, tile33)


def _require_nonzero_ezz(fn_name, tile33):
    if float(np.min(np.abs(tile33[..., 2, 2]))) < 1e-300:
        raise ValueError(f"{fn_name}: e_zz must be nonzero in every region "
                         f"(the E_z elimination divides by it).")


_COMP_IDX = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1),
             "zz": (2, 2), "xz": (0, 2), "yz": (1, 2), "zx": (2, 0),
             "zy": (2, 1)}


def _tile_is_offplane(tile33):
    """True if any region carries out-of-plane coupling (xz/yz/zx/zy)."""
    off = np.abs(tile33[..., [0, 1, 2, 2], [2, 2, 0, 1]])
    return float(np.max(off)) > 0.0


def _assemble_2d_tensor(ax, ay, tile33):
    """Nodal tensor-component operators (k0-free): the multiply-by-component
    Galerkin masses ``C_ab = M^-1 P_ab`` for all NINE components plus the unit
    derivative operators ``Gx0/Gy0 = -i M^-1 D`` (caller divides by ``k0``).
    ``izz`` is the multiply-by-``1/e_zz`` operator (the 'li' E_z
    elimination)."""
    Mx, Dx = ax["M"], ax["D"]
    My, Dy = ay["M"], ay["D"]
    N = ax["n"] * ay["n"]
    M = np.kron(My, Mx)
    DX = np.kron(My, Dx)
    DY = np.kron(Dy, Mx)
    Minv = np.linalg.inv(M)
    Gx0 = -1j * (Minv @ DX)
    Gy0 = -1j * (Minv @ DY)
    P = {k: np.zeros((N, N), dtype=_C) for k in
         tuple(_COMP_IDX) + ("izz",)}
    nsx, nsy = len(ax["Mtile"]), len(ay["Mtile"])
    for sx in range(nsx):
        for sy in range(nsy):
            ker = np.kron(ay["Mtile"][sy], ax["Mtile"][sx])
            t = tile33[sx, sy]
            for k, (a, b) in _COMP_IDX.items():
                P[k] += t[a, b] * ker
            P["izz"] += (1.0 / t[2, 2]) * ker
    ops = {k: Minv @ v for k, v in P.items()}
    return Gx0, Gy0, ops


def _tensor_layer_modes(ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0,
                        ox, oy, kxv, kyv, formulation, return_ops=False):
    """Fourier-basis layer eigenmodes of a full (3, 3) tensor cell -- the
    SEM-projected operators fed to the shared dimension-agnostic
    :func:`_layer_eigenmodes_tensor` (also used per-layer by
    :class:`~lumenairy.elements.pmm.stack2d.PMM2DStack`).  ``tile_i`` is in the
    INTERNAL (conjugated) convention.

    Returns ``(W, V, lam)`` for an IN-PLANE cell, or the GENERATOR 6-tuple
    ``(W, V, lam, Wb, Vb, lam_b)`` when the tile carries out-of-plane coupling
    (xz/yz/zx/zy) -- the caller switches to the generalized S-matrix cascade.

    A UNIFORM cell (no walls) bypasses the SEM grid: the Fourier basis is exact
    there (the convolution of a constant is ``t*I`` and the derivative
    operators are ``diag(k)``), so there is no projection floor -- matching
    rcwa_jones_2d's uniform-cell representation exactly.  A SEPARABLE cell
    (uniform along one axis) gets exact ``diag(k)`` on the wall-less axis.
    """
    Nf = len(kxv)
    nsx, nsy = len(ax["strips"]), len(ay["strips"])
    offp = _tile_is_offplane(tile_i)
    oop = dict(EZX=None, EZY=None, EXZ=None, EYZ=None)
    if offp:
        # ezz-Schur reduction POINTWISE (per region), BEFORE any factorization
        # (Li 2003 / Li 1999 Eq. 12; mirrors rcwa._tensor_convolutions_full):
        # eliminating Ez = (1/ezz)(Dz - ezx Ex - ezy Ey) folds the off-plane
        # coupling into an EFFECTIVE in-plane 2x2 profile a_eff = exx -
        # exz ezx / ezz etc.  Feeding the RAW in-plane components and letting
        # the generator form the Schur composite SPECTRALLY is the wrong
        # factorization order (the "gen2 trap": ~1% eigenvalue error on a
        # uniform medium).  Off-plane + zz components stay raw (the A/B
        # generator cross-blocks + the E_z elimination use them directly).
        te = tile_i.copy()
        izz = tile_i[..., 2, 2]
        for a in (0, 1):
            for b in (0, 1):
                te[..., a, b] = (tile_i[..., a, b]
                                 - tile_i[..., a, 2] * tile_i[..., 2, b] / izz)
        tile_i = te
    if len(x_walls) == 0 and len(y_walls) == 0:
        t0 = tile_i[0, 0]
        I_F = np.eye(Nf, dtype=_C)
        GxF = np.diag(kxv.astype(_C))
        GyF = np.diag(kyv.astype(_C))
        CxxF, CxyF = t0[0, 0] * I_F, t0[0, 1] * I_F
        CyxF, CyyF = t0[1, 0] * I_F, t0[1, 1] * I_F
        EZZ = t0[2, 2] * I_F          # inv-of-inverse == direct for a constant
        if offp:
            oop = dict(EZX=t0[2, 0] * I_F, EZY=t0[2, 1] * I_F,
                       EXZ=t0[0, 2] * I_F, EYZ=t0[1, 2] * I_F)
    elif nsx == 1 or nsy == 1:
        # SEPARABLE cell (uniform along one axis): the wall-less axis is exact
        # diag(k) in the Fourier basis (transverse-momentum conservation EXACT;
        # mirrors twod._scalar_projected_ops) and the patterned axis carries
        # 1-D projected component masses.
        if nsy == 1:
            axd, o_p = ax, ox                     # patterned axis = x
            prof = tile_i[:, 0]                   # (nsx, 3, 3)
        else:
            axd, o_p = ay, oy
            prof = tile_i[0, :]
        M, D = axd["M"], axd["D"]
        Minv = np.linalg.inv(M)
        G0 = -1j * (Minv @ D)
        T1 = _axis_projection(axd, o_p)
        T1p = np.linalg.pinv(T1)
        ip1 = T1 @ T1p
        g1 = T1 @ G0 @ T1p

        def _mass(ab_getter):
            P = np.zeros_like(M)
            for s, Mt in enumerate(axd["Mtile"]):
                P += ab_getter(prof[s]) * Mt
            return T1 @ (Minv @ P) @ T1p

        c = {(a, b): _mass(lambda t, a=a, b=b: t[a, b])
             for a in (0, 1) for b in (0, 1)}
        czz = _mass(lambda t: t[2, 2])
        cizz = _mass(lambda t: 1.0 / t[2, 2])
        ez1 = np.linalg.inv(cizz) if formulation == "li" else czz
        o1 = {}
        if offp:
            o1 = {k: _mass(lambda t, a=a, b=b: t[a, b])
                  for k, (a, b) in (("EZX", (2, 0)), ("EZY", (2, 1)),
                                    ("EXZ", (0, 2)), ("EYZ", (1, 2)))}
        if nsy == 1:
            Iy = np.eye(len(oy), dtype=_C)
            GxF = np.kron(Iy, g1) / k0 + kx0 * np.kron(Iy, ip1)
            GyF = np.diag(kyv.astype(_C))
            CxxF = np.kron(Iy, c[(0, 0)])
            CxyF = np.kron(Iy, c[(0, 1)])
            CyxF = np.kron(Iy, c[(1, 0)])
            CyyF = np.kron(Iy, c[(1, 1)])
            EZZ = np.kron(Iy, ez1)
            if offp:
                oop = {k: np.kron(Iy, v) for k, v in o1.items()}
        else:
            Ix = np.eye(len(ox), dtype=_C)
            GxF = np.diag(kxv.astype(_C))
            GyF = np.kron(g1, Ix) / k0 + ky0 * np.kron(ip1, Ix)
            CxxF = np.kron(c[(0, 0)], Ix)
            CxyF = np.kron(c[(0, 1)], Ix)
            CyxF = np.kron(c[(1, 0)], Ix)
            CyyF = np.kron(c[(1, 1)], Ix)
            EZZ = np.kron(ez1, Ix)
            if offp:
                oop = {k: np.kron(v, Ix) for k, v in o1.items()}
    else:
        # FACTORIZED dense-branch assembly (v5.14 perf audit P1; mirrors
        # twod._scalar_projected_ops): diagonal GLL masses make every tensor
        # component a NODAL VECTOR and the derivative operators kron-factor --
        # no N x N dense materialization.
        Tx = _axis_projection(ax, ox)
        Txp = np.linalg.pinv(Tx)
        Ty = _axis_projection(ay, oy)
        Typ = np.linalg.pinv(Ty)
        mdx = np.diag(ax["M"])
        mdy = np.diag(ay["M"])
        gx1 = Tx @ ((1.0 / mdx)[:, None] * ax["D"]) @ Txp
        gy1 = Ty @ ((1.0 / mdy)[:, None] * ay["D"]) @ Typ
        NxO, NyO = len(ox), len(oy)
        Gx0F = -1j * np.kron(np.eye(NyO, dtype=_C), gx1)
        Gy0F = -1j * np.kron(gy1, np.eye(NxO, dtype=_C))
        Ip = np.kron(Ty @ Typ, Tx @ Txp)
        GxF = Gx0F / k0 + kx0 * Ip
        GyF = Gy0F / k0 + ky0 * Ip
        Mdiag = np.kron(mdy, mdx)
        kers = [[np.kron(np.diag(ay["Mtile"][sy]), np.diag(ax["Mtile"][sx]))
                 for sy in range(nsy)] for sx in range(nsx)]

        def _nodal(getter):
            v = np.zeros_like(Mdiag)
            for sx in range(nsx):
                for sy in range(nsy):
                    v += getter(tile_i[sx, sy]) * kers[sx][sy]
            return v / Mdiag

        Tp = np.kron(Ty, Tx)
        Tpinv = np.kron(Typ, Txp)

        def _proj(getter):
            return (Tp * _nodal(getter)[None, :]) @ Tpinv

        CxxF = _proj(lambda t: t[0, 0])
        CxyF = _proj(lambda t: t[0, 1])
        CyxF = _proj(lambda t: t[1, 0])
        CyyF = _proj(lambda t: t[1, 1])
        if formulation == "li":
            # the shared solver computes Ez_inv = inv(EZZ); feeding inv(EinvF)
            # makes Ez_inv == the projected multiply-by-1/ezz (the hybrid's
            # validated inverse-rule E_z elimination)
            EZZ = np.linalg.inv(_proj(lambda t: 1.0 / t[2, 2]))
        else:
            EZZ = _proj(lambda t: t[2, 2])
        if offp:
            oop = dict(EZX=_proj(lambda t: t[2, 0]),
                       EZY=_proj(lambda t: t[2, 1]),
                       EXZ=_proj(lambda t: t[0, 2]),
                       EYZ=_proj(lambda t: t[1, 2]))
    if return_ops:
        # F2 (audit): expose the projected operators so the even-parity fold
        # can build (P, Q) via rcwa's _tensor_PQ.  Only IN-PLANE cells fold
        # (the out-of-plane generator breaks the +/-lam symmetry) -> None.
        if offp:
            return None
        return GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ
    return _layer_eigenmodes_tensor(GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ,
                                    **oop)


def pmm_jones_2d(
    period_x: float,
    period_y: float,
    eps_tensor_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    n_orders: int = 11,
    formulation: str = "laurent",
    max_nodal_dof: int = _MAX_NODAL_DOF,
    stabilize: bool = False,
    symmetry: bool = False,
):
    """Rigorous 2-D anisotropic grating via the hybrid PMM: a single layer whose
    permittivity is a full IN-PLANE tensor field over an axis-aligned
    piecewise-constant cell.  The PMM mirror of :func:`rcwa_jones_2d` (same
    signature shape, same returns), with the cell walls resolved geometrically
    by spectral elements instead of Fourier sampling.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_tensor_cell : (Sx, Sy, 3, 3) array_like of complex
        Per-pixel permittivity tensor over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  FULL (3, 3) tensors are supported,
        including out-of-plane ``xz/yz/zx/zy`` coupling (the generator path;
        ~8x the in-plane cost -- see the module docstring).  The pixel grid IS
        the geometry (axis-aligned walls derived exactly); see
        :func:`pmm_efficiency_2d_cell`.
    n_substrate, n_superstrate : complex
        Half-space refractive indices (isotropic).
    depth, wavelength : float
        Layer thickness / vacuum wavelength (metres).
    theta, phi : float, optional
        Conical incidence angles (radians).
    degree, elements_per_strip, grade, n_orders, max_nodal_dof
        As in :func:`pmm_efficiency_2d_cell`.
    formulation : {'laurent', 'li'}, optional
        ``E_z``-elimination rule: ``'laurent'`` = ``inv([[e_zz]])`` (the
        :func:`rcwa_jones_2d` mirror; a scalar cell reduces EXACTLY to
        ``pmm_efficiency_2d_cell(formulation='laurent')``); ``'li'`` = the
        projected multiply-by-``1/e_zz`` (the hybrid's inverse-rule
        elimination).  The in-plane tensor block is direct-rule either way
        (see the module docstring for the Li-1997 mixed-rule note).
    stabilize : bool, optional
        Per-order + Jones degree-scan consensus (the 1-D guard against the
        measure-zero quasi-resonances), stepping through consecutive ODD
        degrees.  Expensive in 2-D -> default False.
    symmetry : bool, optional
        Opt-in even-parity fold (audit F2): a centro-symmetric IN-PLANE tensor
        cell at NORMAL incidence excites only even modes, so the single-layer
        solve runs in the ``(Nf+1)``-d even sector (rcwa's :func:`_tensor_PQ`
        folded through :func:`_symmetric_cascade_rt`).  Default off ->
        byte-identical; a per-cell flip-invariance guard falls back to the full
        ``2Nf`` solve (out-of-plane / off-centre / oblique never fold).

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (2, N) float ndarray
        Efficiencies per order; row 0 = incident ``E_x``, row 1 = incident
        ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (PUBLIC ``exp(-i w t)`` convention).
    """
    if formulation not in ("laurent", "li"):
        raise ValueError(
            f"pmm_jones_2d: formulation must be 'laurent' or 'li', got "
            f"{formulation!r}")
    cell = np.asarray(eps_tensor_cell, dtype=_C)
    if cell.ndim != 4 or cell.shape[2:] != (3, 3):
        raise ValueError(
            f"pmm_jones_2d: eps_tensor_cell must be (Sx, Sy, 3, 3), got "
            f"shape {cell.shape}.")
    x_walls, y_walls, tile = _cell_to_walls_tile(
        cell, period_x, period_y, "pmm_jones_2d")
    _require_nonzero_ezz("pmm_jones_2d", tile)

    # Loss-convention bridge: PUBLIC Im(eps)>0 -> internal exp(+iwt); the Jones
    # matrix is conjugated BACK at extraction (the efficiencies are real).
    tile_i = np.conj(tile)
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    def _solve_at(deg):
        return _pmm_jones_2d_at(
            period_x, period_y, x_walls, y_walls, tile_i, eps_sup, eps_sub,
            depth, wavelength, theta, phi, deg, elements_per_strip, grade,
            n_orders, formulation, max_nodal_dof, symmetry)

    if stabilize:
        # consensus over consecutive ODD degrees (the 1-D _stabilize_jones
        # machinery; an unaffordable higher degree ends the scan gracefully)
        return _stabilize_jones(_scan_solver(_solve_at, degree), degree,
                                "pmm_jones_2d",
                                passive_tol=_PASSIVE_TOL_2D,
                                per_order_tol=_PER_ORDER_TOL_2D)
    return _solve_at(degree)


def _pmm_jones_2d_at(period_x, period_y, x_walls, y_walls, tile_i, eps_sup,
                     eps_sub, depth, wavelength, theta, phi, degree,
                     elements_per_strip, grade, n_orders, formulation,
                     max_nodal_dof, symmetry=False):
    """Single fixed-degree tensor solve (eps already internal-convention)."""
    el_x = _axis_elem_counts(period_x, x_walls, degree, elements_per_strip,
                             "pmm_jones_2d", "x")
    el_y = _axis_elem_counts(period_y, y_walls, degree, elements_per_strip,
                             "pmm_jones_2d", "y")
    _validate_cell_orders("pmm_jones_2d", n_orders, degree, el_x, el_y)
    _validate_cell_cost("pmm_jones_2d", el_x, el_y, degree, max_nodal_dof)

    ax = _build_axis(period_x, x_walls, degree, el_x, grade)
    ay = _build_axis(period_y, y_walls, degree, el_y, grade)

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nf = len(order_x)

    # Conical-incidence hardening (mirrors rcwa_jones_2d / the scalar core):
    # reject an evanescent incident wave; nudge off exact Wood anomalies in any
    # constituent medium (the tensor diagonals contribute their real parts).
    _require_propagating_incidence("pmm_jones_2d", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)
    eps_reals = [eps_sup, eps_sub] + [
        complex(e) for e in np.asarray(tile_i[..., [0, 1, 2],
                                              [0, 1, 2]]).ravel()]
    wl = _grazing_safe_wavelength(float(wavelength), kx0, ky0, order_x,
                                  order_y, period_x, period_y, eps_reals)
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)

    # ---- half-space modes (analytic Rayleigh) + layer tensor modes ----------
    Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    orders2d = np.stack([order_x, order_y], axis=1)
    kt = float(np.hypot(kx0, ky0))

    # F2 (audit): even-parity fold of the IN-PLANE tensor layer at normal
    # incidence.  Build (P, Q) via rcwa's _tensor_PQ (byte-identical to the
    # blocks _layer_eigenmodes_tensor eigendecomposes) and run the single-layer
    # cascade in the (Nf+1)-d even sector; None -> not applicable (out-of-plane,
    # off-centre or oblique) -> the full 2Nf solve below (byte-identical there).
    sym_pairs = None
    if symmetry and kt < 1e-12:
        ops = _tensor_layer_modes(
            ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv, kyv,
            formulation, return_ops=True)
        if ops is not None:                        # in-plane only
            from ..rcwa._core import _symmetric_cascade_rt, _tensor_PQ
            GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ = ops
            Pt, Qt = _tensor_PQ(GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ, np)
            sym_pairs = _symmetric_cascade_rt(
                Vsup, Vsub, np.diag(kxv.astype(_C)), np.diag(kyv.astype(_C)),
                [("PQ", Pt, Qt, Cxx)], [depth], k0,
                [np.concatenate([1.0 * delta, 0.0 * delta]),
                 np.concatenate([0.0 * delta, 1.0 * delta])], orders2d, np)

    S11 = S21 = None
    if sym_pairs is None:
        modes = _tensor_layer_modes(
            ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv, kyv,
            formulation)

        if len(modes) == 3:
            # -- in-plane: symmetric +/-lam cascade (the rcwa_jones_2d tail) --
            Wl, Vl, lam_l = modes
            S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
            S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
            S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
        else:
            # -- OUT-OF-PLANE: the full-3x3 generator breaks the [W; -V] <->
            # -lam symmetry, so forward AND backward modes are distinct -> the
            # GENERALIZED S-matrix cascade (the rcwa/oned.py full-3x3 template;
            # the isotropic half-spaces keep their symmetric [W, W; V, -V] form).
            Wf, Vf, lam_f, Wb, Vb, lam_b = modes
            Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
            Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
            Ml = _modes_to_M(Wf, Vf, Wb, Vb)
            S = _interface_smatrix_general(Msup, Ml)
            S = _redheffer_star(
                S, _propagation_smatrix_general(lam_f, lam_b, k0 * depth))
            S = _redheffer_star(S, _interface_smatrix_general(Ml, Msub))
        S11, _S12, S21, _S22 = S

    kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
    kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
    safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    R_rows, T_rows, j_cols = [], [], []
    for ip, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        # Unit tangential E along (ex0, ey0); the incident wave's longitudinal
        # Ez = -(kx0 ex + ky0 ey)/kz_inc inflates |E_inc|^2 (cf. the 1-D sec^2).
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        if sym_pairs is not None:
            r, t = sym_pairs[ip]                   # even-parity fold (F2)
        else:
            cinc = np.concatenate([ex0 * delta, ey0 * delta])
            r = S11 @ cinc
            t = S21 @ cinc
        rx, ry = r[:Nf], r[Nf:]
        tx, ty = t[:Nf], t[Nf:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = np.real(kz_ref_f / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                           + np.abs(rz) ** 2) / einc_sq
        Te = np.real(kz_trn_f / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                           + np.abs(tz) ** 2) / einc_sq
        R_rows.append(np.where(np.real(kz_ref_f) > 0, np.real(Re), 0.0))
        T_rows.append(np.where(np.real(kz_trn_f) > 0, np.real(Te), 0.0))
        # PUBLIC-convention Jones: conjugate back out of the internal gauge
        j_cols.append(np.stack([np.conj(rx[p0]), np.conj(ry[p0])]))
    R_eff = np.stack(R_rows)
    T_eff = np.stack(T_rows)
    jones_reflection = np.stack(j_cols, axis=1)
    return orders2d, R_eff, T_eff, jones_reflection
