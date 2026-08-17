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
At NORMAL incidence on a flip-symmetric cell that ``4*Nf`` eig is replaced by
ONE ``2*Nf`` eig (the parity-sign block reduction; ``symmetry`` on, which is
the default), measured **1.6x - 2.4x** on the whole solve -- see
``docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md``.  Oblique
incidence still pays the full ``4*Nf``.
NON-RECIPROCAL cells (``e_xz != e_zx`` asymmetric, non-Hermitian) can give
``R+T != 1`` PHYSICALLY (no auto-balance) -- match against a 1-D oracle, do
not assert unity.  Loss convention: PUBLIC ``Im(eps) > 0`` for loss (the
conjugation bridge is internal, matching the rest of the suite).
"""
from __future__ import annotations

import numpy as np

from ...backend import is_jax_array
from ..rcwa._core import (
    _check_energy,
    _grazing_safe_wavelength,
    _interface_smatrix_general,
    _layer_eigenmodes_tensor,
    _modes_to_M,
    _norm_slant_pair,
    _oop_block_gauge,
    _propagation_smatrix_general,
    _require_propagating_incidence,
    _slant_is_zero,
    _symmetry_on,
)
from ._core import (
    _interface_smatrix,
    _lossy_incidence,
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


def _require_nonzero_ezz(fn_name, tile33):
    if float(np.min(np.abs(tile33[..., 2, 2]))) < 1e-300:
        raise ValueError(f"{fn_name}: e_zz must be nonzero in every region "
                         f"(the E_z elimination divides by it).")


def _tile_is_offplane(tile33):
    """True if any region carries out-of-plane coupling (xz/yz/zx/zy).

    S1-12 (audit AUDIT_V5_24_2): the test is RELATIVE to the tensor scale,
    not a strict ``> 0.0``.  A cell that is physically in-plane but built
    by rotating a diagonal tensor (or assembled through any float path)
    can carry ~1e-16..1e-17 rounding noise in the xz/yz/zx/zy slots; a
    strict ``> 0`` then mis-routes it to the ~8x-slower 4Nf generalized
    generator and disables the even-parity fold.  A genuine out-of-plane
    coupling is O(tensor scale) (a birefringent tensor tilted by even a
    nano-radian gives off-diagonals many decades above float roundoff),
    so a ``1e-12 * scale`` floor cleanly separates real coupling from
    noise.  ``scale`` is the largest-magnitude tensor component (>= 1 so a
    near-vacuum cell still gets an absolute 1e-12 guard)."""
    off = np.abs(tile33[..., [0, 1, 2, 2], [2, 2, 0, 1]])
    scale = max(float(np.max(np.abs(tile33))), 1.0)
    return float(np.max(off)) > 1e-12 * scale


def _tensor_layer_modes(ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0,
                        ox, oy, kxv, kyv, formulation, return_ops=False,
                        slant=None, block_eig=False):
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
    if formulation == "fff_nv" and offp:
        raise ValueError(
            "pmm_jones_2d: formulation='fff_nv' is IN-PLANE only (the "
            "out-of-plane anisotropic FFF is not implemented) -- use 'laurent' "
            "or 'li' for an out-of-plane tensor cell.")
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
        if formulation == "fff_nv":
            # Full Popov-Neviere OFF-DIAGONAL rule for a SEPARABLE cell: the
            # wall-normal is CONSTANT along the patterned axis (n = e_x for an
            # x-stripe, e_y for a y-stripe), so Q = [[eps.C]][[C]]^-1 reduces
            # EXACTLY to the rigorous Li-1996 1-D anisotropic factorization --
            # the wall-normal diagonal takes the inverse rule and the
            # off-diagonal gets its correct composite (the diagonal-only 'li'
            # leaves Cxy/Cyx Laurent-floored).  Every operator here is a 1-D
            # projected mass along the patterned axis, so the single inversion
            # [[1/e_nn]]^-1 is well-conditioned (no crossed-cell blow-up).  EZZ
            # stays DIRECT (ez1 = czz above; E_z is tangential to every vertical
            # wall -- Li 1997 Eq. 27, the rcwa_jones_2d fff_nv mirror).
            nn = 0 if nsy == 1 else 1          # wall-normal in-plane component
            tt = 1 - nn                        # tangential in-plane component
            inn = np.linalg.inv(_mass(lambda t: 1.0 / t[nn, nn]))
            b_nt = _mass(lambda t: t[nn, tt] / t[nn, nn])
            b_tn = _mass(lambda t: t[tt, nn] / t[nn, nn])
            schur = _mass(lambda t: t[tt, tt] - t[tt, nn] * t[nn, tt] / t[nn, nn])
            c = {(nn, nn): inn,
                 (nn, tt): inn @ b_nt,
                 (tt, nn): b_tn @ inn,
                 (tt, tt): schur + b_tn @ inn @ b_nt}
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
        if formulation == "fff_nv":
            raise ValueError(
                "pmm_jones_2d: formulation='fff_nv' requires a SEPARABLE "
                "(single-orientation, x- or y-patterned) anisotropic cell -- "
                "this cell is patterned along BOTH axes (a varying wall normal), "
                "whose projected anisotropic factorization is ill-conditioned "
                "(the research-grade matched-coordinate FFF regime).  Use "
                "formulation='li' or 'laurent' (both rigorous -- the off-diagonal "
                "stays Laurent-floored but the solve is stable).")
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
        #
        # A SLANTED cell must refuse here for the SAME reason, and the failure
        # mode if it does not is the worst class in this repo's taxonomy.  The
        # fold is taken whenever incidence is normal (kt < 1e-12), and it never
        # reaches _layer_eigenmodes_tensor -- where the slant convection lives.
        # So with the fold left on, a slanted layer SILENTLY RETURNS THE
        # VERTICAL ANSWER: no warning, energy conserved, deterministic, and
        # wrong by 9.2e-02 (10 deg slant) to 2.5e-01 (35 deg), NOT converging
        # with n_orders.  Every OBLIQUE test passes with the bug present, so
        # the gate for this must be a NORMAL-incidence slanted case.
        # MEASURED 2026-08-16; gate: test_slant_normal_incidence_not_folded.
        if offp or not _slant_is_zero(slant):
            return None
        return GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ
    # PARITY-SIGN BLOCK REDUCTION (EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md):
    # the 4Nf generator an OUT-OF-PLANE (or SLANTED) layer needs is
    # block-anti-diagonal under R = diag(I,I,-I,-I).(I4 (x) J) at normal
    # incidence, so ONE 2Nf eig delivers all 4Nf eigenpairs.  This offers the
    # gauge; _generator_block_eig verifies the structure on the assembled
    # generator and falls back to the dense zgeev if it does not hold (oblique
    # incidence, an off-centre cell, an unmirrored wall layout).  The even
    # fold above cannot serve this case -- it needs J ALONE to commute, which
    # the off-plane cross-blocks (linear in K, hence J-ODD) break.
    #
    # OPT-IN per call site (``block_eig``), not on by default: the reduction is
    # validated end to end for the two 2-D entry points that pass it
    # (``pmm_jones_2d`` and ``PMM2DStackHybrid``).  The native-conical and 1-D
    # stack callers reach this same function with a degenerate order table
    # (``oy = [0]``) and are left byte-identical until they get their own
    # two-sided gate tests -- see the doc's S8.
    gauge = None
    if block_eig and (offp or not _slant_is_zero(slant)):
        orders2d = np.stack([np.tile(ox, len(oy)), np.repeat(oy, len(ox))],
                            axis=1)
        gauge = _oop_block_gauge(kxv, kyv, orders2d, CxxF)
    return _layer_eigenmodes_tensor(GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ,
                                    slant=slant, sym_gauge=gauge, **oop)


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
    symmetry="auto",
    region_layout=None,
    slant=None,
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
    formulation : {'laurent', 'li', 'fff_nv'}, optional
        ``E_z``-elimination rule: ``'laurent'`` = ``inv([[e_zz]])`` (the
        :func:`rcwa_jones_2d` mirror; a scalar cell reduces EXACTLY to
        ``pmm_efficiency_2d_cell(formulation='laurent')``); ``'li'`` = the
        projected multiply-by-``1/e_zz`` (the hybrid's inverse-rule
        elimination).  The in-plane tensor block is direct-rule for both of
        those (see the module docstring for the Li-1997 mixed-rule note).
        ``'fff_nv'`` is the full Popov-Neviere (2001) anisotropic OFF-DIAGONAL
        factorization: for a SEPARABLE (single-orientation, e.g. an x- or
        y-patterned stripe) anisotropic cell the wall-normal is constant, so the
        projected tensor operator reduces to the rigorous Li-1996 1-D
        factorization -- the wall-normal diagonal gets the inverse rule and the
        off-diagonal ``Cxy``/``Cyx`` of a rotated director (``exy, eyx != 0``)
        gets its correct composite (the ``'li'`` diagonal rule leaves it
        Laurent-floored).  It converges markedly faster than ``'laurent'`` on
        sharp anisotropic walls and matches the rigorous 1-D solver; NumPy only,
        in-plane only.  A CROSSED (both-axis-patterned) cell has a varying
        normal whose projected factorization is ill-conditioned (the
        matched-coordinate FFF regime) and RAISES -- use ``'li'``/``'laurent'``
        (the :func:`rcwa_jones_2d` mirror gates the same way).
    stabilize : bool, optional
        Per-order + Jones degree-scan consensus (the 1-D guard against the
        measure-zero quasi-resonances), stepping through consecutive ODD
        degrees.  Expensive in 2-D -> default False.
    symmetry : {'auto', True, False}, optional
        Even-parity fold (audit F2): a centro-symmetric IN-PLANE tensor cell at
        NORMAL incidence excites only even modes, so the single-layer solve runs
        in the ``(Nf+1)``-d even sector (rcwa's :func:`_tensor_PQ` folded through
        :func:`_symmetric_cascade_rt`).  ``'auto'`` (the DEFAULT; ``True`` is
        equivalent) folds when the precondition holds; a per-cell
        flip-invariance guard falls back to the full ``2Nf`` solve
        (out-of-plane / off-centre / oblique never fold).  ``symmetry=False``
        forces the full solve (the even basis matches it to ~1e-12, not
        bit-for-bit).

        For an OUT-OF-PLANE (or slanted) tensor cell -- which cannot fold,
        because the off-plane cross-blocks are odd under the order flip -- the
        same setting instead enables the PARITY-SIGN BLOCK REDUCTION of the
        ``4Nf`` generator: at normal incidence on a flip-symmetric cell the
        generator is block-anti-diagonal under the order flip TIMES the E/H
        sign flip, so one ``2Nf`` eig yields all ``4Nf`` eigenpairs (1.6x -
        2.4x on the whole solve).  It is verified on the assembled generator
        every call and falls back to the dense solve otherwise, so it is exact
        or refused; ``symmetry=False`` forces the dense generator.
    region_layout : (Sx, Sy) int array_like, optional
        JAX-only.  A traced ``eps_tensor_cell`` (under ``jax.grad`` /
        ``jax.jit``) cannot define the exact spectral-element walls (that is
        data-dependent control flow), so pass a CONCRETE int grid of the same
        ``(Sx, Sy)`` shape labelling the regions; the traced tensor supplies the
        per-region VALUES (gradients flow through them and through
        depth / wavelength / angles).  See :func:`pmm_efficiency_2d_cell`.  The
        differentiable path always uses the full-3x3 generator (``4*Nf`` eig,
        exact for an in-plane tensor, correct for out-of-plane) so forward and
        gradient share one branch; prefer modest ``n_orders`` for in-plane
        gradient loops.  Cells patterned along BOTH axes only -- a cell uniform
        along an axis (fully uniform, or a 1-D-grating stripe) is degenerate
        under ``jnp.linalg.eig`` and RAISES (use the differentiable
        ``pmm_jones_1d`` / ``rcwa_jones_1d`` for a 1-D grating, or
        ``berreman_jones_1d`` for a uniform anisotropic layer).  Ignored on the
        NumPy path.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (2, N) float ndarray
        Efficiencies per order; row 0 = incident ``E_x``, row 1 = incident
        ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (PUBLIC ``exp(-i w t)`` convention; columns = response to incident
        ``E_x`` / ``E_y``).  This is the CARTESIAN basis; the 1-D solvers
        return ``te``/``tm`` (``s``/``p``), which coincides with ``(x, y)``
        (``tm`` <-> ``x``, ``te`` <-> ``y``) only at ``phi = 0``.  For
        conical incidence (``phi != 0``) the two differ by the rotation
        into the plane of incidence.  See CONVENTIONS.md sec 7.1.

    Notes
    -----
    Incidence is set by the conical pair ``theta`` (polar) / ``phi``
    (azimuth), both radians.  There is NO ``angle`` keyword on this 2-D
    entry (the 1-D solvers' ``angle``/``theta`` alias does not apply here)
    -- passing ``angle=`` raises ``TypeError``.  Use ``theta``.
    """
    if formulation not in ("laurent", "li", "fff_nv"):
        raise ValueError(
            f"pmm_jones_2d: formulation must be 'laurent', 'li' or 'fff_nv', "
            f"got {formulation!r}")
    if formulation == "fff_nv" and any(is_jax_array(a) for a in
                                       (eps_tensor_cell, n_substrate,
                                        n_superstrate, depth, wavelength,
                                        theta, phi)):
        raise ValueError(
            "pmm_jones_2d: formulation='fff_nv' is NumPy only -- use 'laurent' "
            "or 'li' on the JAX backend.")

    # ---- JAX (differentiable) dispatch --------------------------------------
    # A traced eps_tensor_cell / source value routes to the full-3x3-generator
    # twin.  This branch MUST precede the np.asarray coercion below, which would
    # sever the JAX trace (materialize a tracer -> TracerArrayConversionError).
    _jx = (eps_tensor_cell, n_substrate, n_superstrate, depth, wavelength,
           theta, phi)
    if any(is_jax_array(a) for a in _jx):
        if stabilize:
            raise ValueError(
                "pmm_jones_2d: stabilize=True is not differentiable "
                "(host-side degree-scan consensus); pass stabilize=False on "
                "the JAX path.")
        if region_layout is None:
            raise ValueError(
                "pmm_jones_2d: a traced eps_tensor_cell cannot define the "
                "exact walls -- pass region_layout (a CONCRETE int grid of the "
                "same (Sx, Sy) shape labelling the regions) on the JAX path.")
        lay = np.ascontiguousarray(np.asarray(region_layout))
        if lay.ndim != 2 or lay.shape != tuple(np.shape(eps_tensor_cell)[:2]):
            raise ValueError(
                f"pmm_jones_2d: region_layout must be a 2-D int grid of shape "
                f"{tuple(np.shape(eps_tensor_cell)[:2])} (the (Sx, Sy) of "
                f"eps_tensor_cell), got shape {lay.shape}.")
        # honour max_nodal_dof on the JAX branch too (audit B1 parity): compute
        # the wall / element counts from the concrete region_layout and reject
        # too-large cells before _static_prep_cell hits a dense assembly.
        xw, yw, _ = _cell_to_walls_tile(
            lay.astype(complex), period_x, period_y, "pmm_jones_2d")
        if len(xw) == 0 or len(yw) == 0:
            # A cell that is UNIFORM along an axis (no wall on x OR on y --
            # including a fully uniform single-region cell and a 1-axis-patterned
            # "stripe") leaves that axis's Fourier orders UNCOUPLED, so the modal
            # spectrum is degenerate and jnp.linalg.eig returns an
            # ill-conditioned eigenbasis (~1e-3 error) -- the same degeneracy the
            # symmetric jax branch cures with an iso-uniform blend, which the
            # full-3x3 generator lacks.  A cell patterned along BOTH axes couples
            # every order, lifts the degeneracy, and solves to machine precision.
            # A cell uniform along one axis is a 1-D grating (use the
            # differentiable pmm_jones_1d / rcwa_jones_1d); a fully uniform
            # anisotropic layer is a planar problem (use the differentiable
            # berreman_jones_1d, exact and analytic).
            raise NotImplementedError(
                "pmm_jones_2d: the JAX (differentiable) path requires a cell "
                "patterned along BOTH axes (region_layout must vary in x AND "
                "y); a cell uniform along an axis has a degenerate modal "
                "spectrum that makes jnp.linalg.eig ill-conditioned (~1e-3 "
                "error).  For a 1-D grating (uniform along one axis) use the "
                "differentiable pmm_jones_1d / rcwa_jones_1d; for a uniform "
                "anisotropic layer use the differentiable berreman_jones_1d "
                "(exact, analytic).")
        elx = _axis_elem_counts(period_x, xw, degree, elements_per_strip,
                                "pmm_jones_2d", "x")
        ely = _axis_elem_counts(period_y, yw, degree, elements_per_strip,
                                "pmm_jones_2d", "y")
        _validate_cell_orders("pmm_jones_2d", n_orders, degree, elx, ely)
        _validate_cell_cost("pmm_jones_2d", elx, ely, degree, max_nodal_dof)
        from ._jax_twod_jones import _pmm_jones_2d_cell_jax
        return _pmm_jones_2d_cell_jax(
            period_x, period_y, eps_tensor_cell, region_layout, n_substrate,
            n_superstrate, depth, wavelength, theta=theta, phi=phi,
            degree=degree, elements_per_strip=elements_per_strip, grade=grade,
            n_orders=n_orders, formulation=formulation)

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
    slant = _norm_slant_pair(slant, "pmm_jones_2d")
    tile_i = np.conj(tile)
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    def _solve_at(deg):
        return _pmm_jones_2d_at(
            period_x, period_y, x_walls, y_walls, tile_i, eps_sup, eps_sub,
            depth, wavelength, theta, phi, deg, elements_per_strip, grade,
            n_orders, formulation, max_nodal_dof, symmetry, slant)

    if stabilize:
        # consensus over consecutive ODD degrees (the 1-D _stabilize_jones
        # machinery; an unaffordable higher degree ends the scan gracefully)
        return _stabilize_jones(_scan_solver(_solve_at, degree), degree,
                                "pmm_jones_2d",
                                passive_tol=_PASSIVE_TOL_2D,
                                per_order_tol=_PER_ORDER_TOL_2D,
                                super_unity_ok=_lossy_incidence(n_superstrate))
    res = _solve_at(degree)
    # Blowup guard (mirror rcwa_jones_2d): a high-contrast / birefringent cell
    # at a near-singular (degree, n_orders) truncation -- common at CONICAL
    # incidence -- can return a NON-PHYSICAL answer (sum R+T up to ~1e7) with no
    # signal.  _check_energy RAISES on that catastrophic case and WARNS on a
    # lossless-closure violation, both pointing at ``stabilize=True`` (which
    # retries nearby degrees and cures it -- verified on birefringent conical
    # gratings).  Provably lossless == every cell permittivity exactly real.
    lossless = not bool(np.any(np.abs(np.imag(np.asarray(tile, dtype=_C)))
                               > 1e-12))
    _check_energy("pmm_jones_2d", res[1], res[2], lossless=lossless)
    return res


def _pmm_jones_2d_at(period_x, period_y, x_walls, y_walls, tile_i, eps_sup,
                     eps_sub, depth, wavelength, theta, phi, degree,
                     elements_per_strip, grade, n_orders, formulation,
                     max_nodal_dof, symmetry=False, slant=None):
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
    _sym = _symmetry_on(symmetry)
    if _sym and kt < 1e-12 and formulation != "fff_nv":
        ops = _tensor_layer_modes(
            ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv, kyv,
            formulation, return_ops=True, slant=slant)
        if ops is not None:                        # in-plane, UNSLANTED only
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
            formulation, slant=slant, block_eig=_sym)

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
