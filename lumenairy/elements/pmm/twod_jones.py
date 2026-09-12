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
``1/e_zz`` directly.  NB Li 1997 Eqs. (8)/(9) + (31) show the OPTIMAL
crossed-grating rule gives each diagonal slot the inverse rule along its own
axis and Laurent along the other (the mixed composites); that per-direction
refinement is NOT implemented -- patterned tensor cells converge at the Laurent
(~1e-3) floor.

WHICH RULE TO ASK FOR.  Because ``'li'`` and ``'laurent'`` differ ONLY in the
``E_z`` rule here, ``'li'`` is not the "more rigorous" option it is on the
scalar entries, and it is measurably the WORSE one: on the reflection Jones of
a high-contrast Si stripe at ``n_orders = 11`` the error against a converged
1-D Li oracle is ``2.7e-03`` (``'fff_nv'``) / ``3.0e-02`` (``'laurent'``) /
``6.9e-02`` (``'li'``), and on a C4 Si pillar ``'li'`` keeps the wrong SIGN on
``Im(Jxx)`` at every affordable truncation (``arg(Jxx)`` 8.9 deg out against
1.2 deg for ``'laurent'``) while energy closes to 1e-5 on both.  Prefer
``formulation='auto'`` -- ``'fff_nv'`` on a separable in-plane cell, where it
IS the rigorous 1-D anisotropic factorization, and ``'laurent'`` otherwise.

ACCURACY CEILING.  This entry is the Fourier-projected HYBRID, so its lossless
energy closure PLATEAUS at ~1e-4..1e-3 and is not monotone in ``n_orders``
(measured on a lossless Si pillar: 1.5e-03 / 1.4e-04 / 5.9e-04 at
``n_orders`` = 5 / 9 / 11).  For energy- or phase-critical work use the
no-floor staggered siblings
:func:`~lumenairy.elements.pmm.twod_staggered.pmm_jones_2d_staggered` /
:class:`~lumenairy.elements.pmm.stack2d_pure.PMM2DStackPure`, whose closure is
``n_orders``-INVARIANT to 14 digits and improves spectrally with ``n_modes``
(2.9e-06 / 1.8e-10 / 1.1e-12 at M = 5 / 8 / 10 on a comparable pillar) -- at a
wall-time price (712-911 s for the M = 10 solve against 7.3 s for the hybrid at
``n_orders = 11``).  Watching ``sum(R)+sum(T)`` alone is NOT a convergence
proof on this engine: see :func:`~lumenairy.elements.pmm.twod.pmm_2d_order_drift`.

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
                        slant=None, block_eig=False, keep=None):
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

    ``keep`` is the Lalanne-1997 CIRCULAR-truncation boolean mask over the full
    rectangular box (``kxv``/``kyv`` must be the FULL box, since the separable
    and crossed branches build ``kron``-factored operators whose shape is
    ``len(ox) * len(oy)``).  Every operator is assembled on the box and then
    restricted by ``np.ix_(keep, keep)`` -- which is exactly the operator built
    on the circular order list, because these are all functions of the order
    LIST -- so the eig runs at ``~(pi/4) Nf``.  The scalar core does the same
    thing to ``lops`` (``twod._pmm2d_solve_core``).
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
        # The GLL mass and every ``Mtile`` are EXACTLY diagonal (see
        # twod._axis_ops_1d), so the nodal inverse is a reciprocal and a
        # component mass accumulates as a nodal VECTOR: this drops one
        # ``O(n^3)`` LAPACK inversion and one dense ``n x n`` matmul PER TENSOR
        # COMPONENT (there are four to eight of them per layer).  BIT-IDENTICAL
        # -- ``np.diag(minv * p)`` is ``np.array_equal`` to ``inv(M) @ P`` and
        # the ``T1 @ ... @ T1p`` sandwich is unchanged.  (Collapsing that
        # sandwich to ``(T1 * v) @ T1p`` as well would be cheaper again but is
        # only 1 ULP identical -- 1.75e-16 relative, measured -- so it is not
        # taken on this validated path.)  This SEPARABLE branch is the one a
        # 1-D grating layer in a 2-D stack takes, i.e. the LC-QWP geometry.
        md = np.diag(M)
        minv = 1.0 / md
        G0 = -1j * (minv[:, None] * D)
        T1 = _axis_projection(axd, o_p)
        T1p = np.linalg.pinv(T1)
        ip1 = T1 @ T1p
        g1 = T1 @ G0 @ T1p
        mtiles = [np.diag(Mt) for Mt in axd["Mtile"]]

        def _mass(ab_getter):
            p = np.zeros_like(md)
            for s, mt in enumerate(mtiles):
                p += ab_getter(prof[s]) * mt
            return T1 @ np.diag(minv * p) @ T1p

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
    if keep is not None:
        ix = np.ix_(keep, keep)
        GxF, GyF = GxF[ix], GyF[ix]
        CxxF, CxyF = CxxF[ix], CxyF[ix]
        CyxF, CyyF = CyxF[ix], CyyF[ix]
        EZZ = EZZ[ix]
        oop = {kk: (None if v is None else v[ix]) for kk, v in oop.items()}
        Nf = int(np.count_nonzero(keep))
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
        if keep is not None:              # the gauge lives on the RETAINED set
            orders2d = orders2d[keep]
            kxv, kyv = kxv[keep], kyv[keep]
        gauge = _oop_block_gauge(kxv, kyv, orders2d, CxxF)
    return _layer_eigenmodes_tensor(GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ,
                                    slant=slant, sym_gauge=gauge, **oop)


#: Absolute tolerance on "is this cell CONSTANT-valued?", mirroring
#: :func:`lumenairy.elements.pmm.stack2d._layer_enters_slant_frame`'s own
#: uniform-tile test (``max|tile - tile.flat[0]| < 1e-12``) so the two entry
#: points can never disagree about which slanted cells are genuine no-ops.
_SLANT_NOOP_TOL = 1e-12


def _slanted_cell_is_a_frame_noop(cell):
    """Is a slanted ``eps_tensor_cell`` a genuine NO-OP -- i.e. CONSTANT-valued
    over the cell, so the shear is a pure coordinate change of a UNIFORM
    medium and the lab answer is the vertical film's?

    MEASURED on a constant-tile ``(6, 4, 3, 3)`` cell at ``slant = (0.5, 0)``,
    oblique 25, against the same call with ``slant=None``: ``dR 4.9e-17 /
    dT 2.1e-14 / dJones 5.4e-16`` (WIN) and ``8.3e-17 / 2.4e-14 / 2.1e-16``
    (WSL) -- a machine-precision no-op on all four returns, which is why such a
    cell must keep SOLVING on the JAX dispatch below rather than be swept up by
    the refusal.

    A TRACED cell cannot be inspected (that is data-dependent control flow), so
    it answers ``False`` -- the same "be loud, not silently anchor-free" policy
    ``_layer_enters_slant_frame`` takes for its own traced tiles.
    """
    if is_jax_array(cell):
        return False
    c = np.asarray(cell)
    if c.size == 0:
        return False
    return bool(np.all(np.abs(c - c.reshape(-1, *c.shape[2:])[0])
                       < _SLANT_NOOP_TOL))


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
    truncation: str = "rectangular",
    max_nodal_dof: int = _MAX_NODAL_DOF,
    stabilize: bool = False,
    symmetry="auto",
    region_layout=None,
    slant=None,
    return_jones_transmission: bool = False,
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
    formulation : {'laurent', 'li', 'fff_nv', 'auto'}, optional
        ``E_z``-elimination rule.  ``'laurent'`` (the DEFAULT) =
        ``inv([[e_zz]])`` -- the :func:`rcwa_jones_2d` mirror; a scalar cell
        reduces EXACTLY to ``pmm_efficiency_2d_cell(formulation='laurent')``.
        ``'li'`` = the projected multiply-by-``1/e_zz``.  On THIS entry the
        two differ ONLY in that ``E_z`` rule: the in-plane tensor block is
        Laurent either way (see the module docstring for the Li-1997
        mixed-rule note), so ``'li'`` is NOT the wall-normal inverse rule it
        is on the scalar entries, and it is not the more rigorous choice
        here.

        MEASURED, and the reason to read this paragraph before picking
        ``'li'`` for rigour.  High-contrast separable Si stripe (eps 12.25,
        duty 0.5, ``Lambda = 0.47 um``, ``lambda = 1 um``, ``d = 0.3 um``,
        ``n_sub = 1.5``, normal incidence, degree 11), reflection ``Jxx``
        against ``rcwa_jones_1d(n_orders=80, 'li')``:

        ==========  ==========  ==========  ==========  ================
        n_orders    'fff_nv'    'laurent'   'li'        rcwa_jones_2d li
        ==========  ==========  ==========  ==========  ================
        5           4.9e-02     1.4e-01     1.8e-01     7.3e-03
        11          2.7e-03     3.0e-02     6.9e-02     2.9e-04
        ==========  ==========  ==========  ==========  ================

        On a C4 Si pillar (crossed, so ``'fff_nv'`` is unavailable) ``'li'``
        additionally keeps the WRONG SIGN on ``Im(Jxx)`` at every affordable
        truncation -- ``arg(Jxx)`` is 8.9 deg out at ``n_orders = 11``
        against 1.2 deg for ``'laurent'`` -- while energy closes to 1e-5 on
        both, so no tripwire fires.  Order of preference ON THE REFLECTION
        JONES, which is what both tables above measure: ``'fff_nv'`` where it
        is available, then ``'laurent'``, then ``'li'``.

        THE TRANSMISSION RETARDANCE RANKS DIFFERENTLY, and it is the
        observable a waveplate is designed against (``return_jones_transmission``
        below).  On the form-birefringent Si/air stripe at ``Lambda/lambda =
        0.2`` (duty 0.5, ``d = 208.14 nm``), against the
        ``rcwa_jones_1d(n_orders=60, 'li')`` reference ``+100.066 deg`` and a
        no-floor ``PMM2DStackPure`` cross-check (``+100.05 deg`` at M = 8),
        ``wrap(arg(J^t_yy) - arg(J^t_xx))`` reads

        ==========  ==========  ==========  ==========
        n_orders    'fff_nv'    'laurent'   'li'
        ==========  ==========  ==========  ==========
        5           +100.12     +94.69      +96.73
        9           +100.23     +98.94      +100.04
        15          +99.90      +99.63      +100.04
        ==========  ==========  ==========  ==========

        -- so ``'li'`` is the MOST accurate of the three at ``n_orders >= 9``
        here (0.03 deg, against 0.17-0.34 deg for ``'fff_nv'``), even though it
        is the worst on the reflection Jones of the stripe above.  Pick the
        rule for the observable you are designing against, and measure it:
        ``'fff_nv'`` is the safe default for reflection and for a coarse
        truncation, ``'li'`` is worth checking for a converged transmission
        phase.

        ``'auto'`` picks the REFLECTION ordering for you: ``'fff_nv'`` on a
        SEPARABLE in-plane
        cell (where it is both available and best) and ``'laurent'``
        otherwise.  It is not the default only because the default must keep
        the exact ``pmm_efficiency_2d_cell('laurent')`` reduction above;
        ``'auto'`` is the recommended setting for new code.  On the JAX path
        ``'auto'`` resolves to ``'laurent'`` (``'fff_nv'`` is NumPy only).

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
        (the :func:`rcwa_jones_2d` mirror gates the same way).  It takes the
        even-parity fold like the other rules (measured fold-vs-full
        ``|dJ| = 1.5e-12`` at degree 11 / ``n_orders = 11``, worth 3.96x).
    truncation : {'rectangular', 'circular'}, optional
        Order-set shape.  ``'circular'`` is the Lalanne-1997 truncation: keep
        only the orders inside the largest reciprocal-space circle inscribed in
        the ``n_orders`` box, which is isotropic in resolution and drops the
        wasted high-``|G|`` corners -- ``Nf -> ~(pi/4) Nf`` and the ``O(Nf^3)``
        eig ~x0.48.  It reached only ``pmm_efficiency_2d[_cell]`` before; the
        operators are functions of the order LIST, so restricting them to the
        circular subspace IS the operator built on that subspace.  NumPy only
        (the jnp twins keep the rectangular box).
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
    return_jones_transmission : bool, optional
        Also return the zeroth-order TRANSMISSION Jones as a FIFTH element
        (default ``False`` keeps the released 4-tuple) -- the
        :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` keyword of the same
        name, on the same contract.  NumPy only (the jnp twin keeps no
        amplitudes); with ``stabilize=True`` it is the transmission Jones of
        the degree the consensus picked.

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
    jones_transmission : (2, 2) complex ndarray
        ONLY when ``return_jones_transmission=True``: the zeroth-order Jones
        TRANSMISSION matrix -- same basis, same convention, same column
        meaning as ``jones_reflection``.

    Notes
    -----
    Incidence is set by the conical pair ``theta`` (polar) / ``phi``
    (azimuth), both radians.  There is NO ``angle`` keyword on this 2-D
    entry (the 1-D solvers' ``angle``/``theta`` alias does not apply here)
    -- passing ``angle=`` raises ``TypeError``.  Use ``theta``.

    **Which observable a transmissive metasurface needs.**  For a transmissive
    waveplate / modulator the phase observable is the TRANSMISSION Jones, not
    the reflection one: pass ``return_jones_transmission=True`` here for a
    single layer, or use
    :meth:`~lumenairy.elements.pmm.stack2d.PMM2DStackHybrid.jones_transmission`
    /
    :meth:`~lumenairy.elements.pmm.stack2d_pure.PMM2DStackPure.jones_transmission`
    for a multilayer stack.  All three carry ONE convention -- rows
    ``[E_x; E_y]``, columns = incident ``E_x``/``E_y``, PUBLIC
    ``exp(-i w t)`` -- so they are interchangeable and drop into a
    :class:`~lumenairy.polarization.JonesField` pipeline UNCONJUGATED.
    MEASURED on a form-birefringent Si/air grating at ``Lambda/lambda = 0.2``
    (duty 0.5, ``d = lambda/4/(n_par - n_perp) = 208.14 nm``, slow axis along
    the grooves): the retardance ``wrap(arg(J^t_yy) - arg(J^t_xx))`` is
    POSITIVE on the SLOW axis -- the slow axis carries
    ``exp(+i * retardance)``, exactly CONVENTIONS.md sec 7 -- and reads
    ``+100.24 deg`` at ``n_orders = 15`` against the
    ``rcwa_jones_1d(n_orders=60, 'li')`` reference ``+100.066 deg``
    (``+0.18 deg`` out); ``n_orders = 9`` is still ``1.40 deg`` out and a
    0th-order Rytov EMT slab is ``4.19 deg`` out.  For a purely 1-D grating
    prefer :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` or the no-floor
    :func:`~lumenairy.elements.pmm.pmm_jones_1d` -- both are ~10x more
    accurate than this entry's ``'fff_nv'`` and ~250x more than its ``'li'``
    at equal ``n_orders``.
    """
    if formulation not in ("laurent", "li", "fff_nv", "auto"):
        raise ValueError(
            f"pmm_jones_2d: formulation must be 'laurent', 'li', 'fff_nv' or "
            f"'auto', got {formulation!r}")
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
    if truncation not in ("rectangular", "circular"):
        raise ValueError(
            f"pmm_jones_2d: truncation must be 'rectangular' or 'circular', "
            f"got {truncation!r}")
    if any(is_jax_array(a) for a in _jx) and truncation != "rectangular":
        raise NotImplementedError(
            "pmm_jones_2d: truncation='circular' is NumPy only -- the jnp twin "
            "(_pmm_jones_2d_cell_jax) builds its order set and its frozen "
            "projectors on the full rectangular box.  Use NumPy inputs, or "
            "truncation='rectangular' on the JAX path.")
    if any(is_jax_array(a) for a in _jx) and return_jones_transmission:
        raise NotImplementedError(
            "pmm_jones_2d: return_jones_transmission=True is NumPy only -- the "
            "jnp twin (_pmm_jones_2d_cell_jax) returns efficiencies and the "
            "reflection Jones and keeps no transmitted amplitudes.  Use NumPy "
            "inputs, or differentiate rcwa_jones_1d / pmm_jones_1d "
            "(return_jones_transmission=True) for a 1-D grating.")
    if any(is_jax_array(a) for a in _jx):
        # 'fff_nv' is NumPy only, so 'auto' resolves to the better of the two
        # rules the jnp twin implements (see the formulation docstring's
        # measured table).
        if formulation == "auto":
            formulation = "laurent"
        if stabilize:
            raise ValueError(
                "pmm_jones_2d: stabilize=True is not differentiable "
                "(host-side degree-scan consensus); pass stabilize=False on "
                "the JAX path.")
        # SLANT (2026-09-11, defect V1 of
        # docs/audits/VERIFY_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md).
        # This branch NEVER READ ``slant``: it hands off to
        # ``_pmm_jones_2d_cell_jax`` without it, and the
        # ``slant = _norm_slant_pair(...)`` normalization below is not even
        # reached until after the branch.  The jnp twin takes no ``slant``
        # argument at all (``_pmm_jones_2d_cell_jax``'s signature has none, and
        # before this fix the string did not occur in that module once), so
        # EVERY ONE of the seven members of ``_jx`` silently returned the
        # VERTICAL answer.  MEASURED before this guard (jax 0.11.0 / 0.10.2, x64), a
        # slanted patterned (6, 4, 3, 3) cell at slant = (0.5, 0), oblique 25,
        # n_orders 3, degree 5, all seven traced routes reading identically:
        # ``dR 1.3e-15 / dT 4.2e-15 / dJones 1.5e-14`` against the NumPy
        # VERTICAL call and ``dR 3.670e-03 / dT 5.514e-02 / dJones 1.546e-02``
        # against the correct NumPy SLANTED one, with ZERO warnings -- i.e. it
        # WAS the vertical answer, not merely close to it.  Unlike the 2-D
        # hybrid's frame anchor this is not a unimodular phase: ``R`` and ``T``
        # are wrong here too, so nothing protects it.
        #
        # Same shape, and same remedy, as the ``PMM2DStackHybrid.solve``
        # refusal (stack2d.py, 2026-09-11): the decision is
        # "does the cell actually get SOLVED IN A SHEARED FRAME", so a
        # CONSTANT-tile slanted cell -- a measured no-op, see
        # :func:`_slanted_cell_is_a_frame_noop` -- and every VERTICAL traced
        # call still solve, bit for bit.
        if not _slant_is_zero(slant) and not _slanted_cell_is_a_frame_noop(
                eps_tensor_cell):
            _traced = ", ".join(
                nm for nm, a in zip(
                    ("eps_tensor_cell", "n_substrate", "n_superstrate",
                     "depth", "wavelength", "theta", "phi"), _jx)
                if is_jax_array(a))
            raise NotImplementedError(
                f"pmm_jones_2d: a SLANTED patterned cell is not "
                f"differentiable -- the traced input(s) [{_traced}] route this "
                f"call to the jnp twin (_pmm_jones_2d_cell_jax), which has no "
                f"notion of a slant and would SILENTLY return the VERTICAL "
                f"answer (measured: dR 1.3e-15 / dT 4.2e-15 / dJones 1.5e-14 "
                f"against the NumPy VERTICAL call, dR 3.7e-03 / dT 5.5e-02 / "
                f"dJones 1.5e-02 against the correct NumPy SLANTED one, with "
                f"no warning).  Use NumPy inputs for slanted cells, or "
                f"z-staircase the slanted layer into vertical ones "
                f"(PMMStack.add_tapered_grating / PMM2DStackHybrid).  A "
                f"CONSTANT-valued cell is a genuine no-op and still solves.")
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

    if formulation == "auto":
        # 'fff_nv' exactly where it is BOTH available and best: a SEPARABLE
        # (one wall-less axis) IN-PLANE cell.  A cell with no walls at all is
        # uniform, where all three rules coincide and 'laurent' is the cheaper
        # branch (no [[1/e_nn]] inversion); a CROSSED or out-of-plane cell
        # cannot take 'fff_nv' at all.  Measured ranking behind this choice:
        # see the ``formulation`` parameter docstring.
        _wall_less = (len(x_walls) == 0, len(y_walls) == 0)
        formulation = ("fff_nv" if (any(_wall_less) and not all(_wall_less)
                                    and not _tile_is_offplane(tile))
                       else "laurent")

    # Loss-convention bridge: PUBLIC Im(eps)>0 -> internal exp(+iwt); the Jones
    # matrix is conjugated BACK at extraction (the efficiencies are real).
    slant = _norm_slant_pair(slant, "pmm_jones_2d")
    tile_i = np.conj(tile)
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    # The transmission Jones rides along on the 5th slot of every solve; the
    # 4-tuple contract is restored at the two return sites below.  ``_jt_seen``
    # pairs each scanned degree's reflection-Jones OBJECT with its transmission
    # partner so the stabilize consensus -- which returns the very array it was
    # handed -- can hand back the matching transmission Jones by identity.
    _jt_seen = []

    def _solve_at(deg):
        out = _pmm_jones_2d_at(
            period_x, period_y, x_walls, y_walls, tile_i, eps_sup, eps_sub,
            depth, wavelength, theta, phi, deg, elements_per_strip, grade,
            n_orders, formulation, max_nodal_dof, symmetry, slant,
            truncation=truncation)
        _jt_seen.append((out[3], out[4]))
        return out[:4]

    if stabilize:
        # consensus over consecutive ODD degrees (the 1-D _stabilize_jones
        # machinery; an unaffordable higher degree ends the scan gracefully)
        res = _stabilize_jones(_scan_solver(_solve_at, degree), degree,
                               "pmm_jones_2d",
                               passive_tol=_PASSIVE_TOL_2D,
                               per_order_tol=_PER_ORDER_TOL_2D,
                               super_unity_ok=_lossy_incidence(n_superstrate))
        if not return_jones_transmission:
            return res
        for _jr, _jt in _jt_seen:
            if _jr is res[3]:
                return tuple(res) + (_jt,)
        raise AssertionError(                          # pragma: no cover
            "pmm_jones_2d: the stabilize consensus returned a Jones matrix "
            "that no scanned degree produced.")
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
    if return_jones_transmission:
        return tuple(res) + (_jt_seen[-1][1],)
    return res


def _pmm_jones_2d_at(period_x, period_y, x_walls, y_walls, tile_i, eps_sup,
                     eps_sub, depth, wavelength, theta, phi, degree,
                     elements_per_strip, grade, n_orders, formulation,
                     max_nodal_dof, symmetry=False, slant=None,
                     truncation="rectangular"):
    """Single fixed-degree tensor solve (eps already internal-convention).

    Returns ``(orders, R_eff, T_eff, jones_reflection, jones_transmission)``;
    the public entry trims the 5th element unless it was asked for."""
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
    # F8: Lalanne-1997 CIRCULAR truncation -- the same mask the scalar core
    # applies to ``lops``.  The kron-factored tensor operators are built on the
    # FULL box inside _tensor_layer_modes and restricted there by ``keep``.
    keep = None
    if truncation == "circular":
        gx_o = order_x / float(period_x)
        gy_o = order_y / float(period_y)
        r2 = min(n_orders / float(period_x), n_orders / float(period_y)) ** 2
        keep = (gx_o ** 2 + gy_o ** 2) <= r2 * (1.0 + 1e-9)
        order_x, order_y = order_x[keep], order_y[keep]
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
                                  order_y, period_x, period_y, eps_reals,
                                  fn_name="pmm_jones_2d")
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)
    # the layer operators are assembled on the FULL box (they kron-factor) and
    # restricted by ``keep`` inside _tensor_layer_modes
    kxv_box = (kxv if keep is None
               else kx0 + np.tile(ox, len(oy)) * (wl / period_x))
    kyv_box = (kyv if keep is None
               else ky0 + np.repeat(oy, len(ox)) * (wl / period_y))

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
    # ``fff_nv`` folds too.  It is reachable ONLY on a SEPARABLE cell (the
    # crossed branch raises), where the wall normal is constant and every
    # operator the fold touches -- ``Cxx/Cxy/Cyx/Cyy`` and ``EZZ`` -- is a 1-D
    # projected mass along the patterned axis kron'd with an identity, exactly
    # the shape the 'li'/'laurent' branches hand over.  The fold's own
    # precondition (a centro-symmetric cell at normal incidence) is unchanged
    # and still auto-detected, so this only stops the BEST formulation from
    # also being the slowest.
    sym_pairs = None
    _sym = _symmetry_on(symmetry)
    if _sym and kt < 1e-12:
        ops = _tensor_layer_modes(
            ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv_box,
            kyv_box, formulation, return_ops=True, slant=slant, keep=keep)
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
            ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv_box,
            kyv_box, formulation, slant=slant, block_eig=_sym, keep=keep)

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
    # FRAME ANCHOR for the TRANSMITTED amplitudes of a genuinely sheared cell
    # -- the stack's :func:`stack2d._slant_frame_walk` reduced to one layer.
    # The sheared frame ``u = x - t z`` is anchored at the layer's TOP face, so
    # its exit plane sits a lateral ``t * depth`` from the lab one and each
    # transmitted order picks up ``exp(+i k0 (alpha_m . t) d)``.  It is a
    # unimodular per-order phase: ``R``, ``T`` and the REFLECTION Jones never
    # see it (they are computed from the untouched cascade output), which is
    # exactly why its absence in the stack was invisible to every energy check.
    # A CONSTANT-valued cell is a pure coordinate change of a uniform medium
    # (:func:`_slanted_cell_is_a_frame_noop`) and takes no anchor, mirroring
    # ``stack2d._layer_enters_slant_frame``.
    tphase = None
    if not _slant_is_zero(slant) and not _slanted_cell_is_a_frame_noop(tile_i):
        tphase = np.exp(1j * k0 * (kxv * float(slant[0]) * float(depth)
                                   + kyv * float(slant[1]) * float(depth)))
    R_rows, T_rows, j_cols, jt_cols = [], [], [], []
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
        # ... and the TRANSMISSION twin, in the SAME layout (rows [Ex; Ey],
        # columns = incident pol) that PerOrderAmplitudesMixin.jones_transmission
        # returns for the two stacks, plus the sheared-frame anchor.
        tx0, ty0 = np.conj(tx[p0]), np.conj(ty[p0])
        if tphase is not None:
            tx0, ty0 = tx0 * tphase[p0], ty0 * tphase[p0]
        jt_cols.append(np.stack([tx0, ty0]))
    R_eff = np.stack(R_rows)
    T_eff = np.stack(T_rows)
    jones_reflection = np.stack(j_cols, axis=1)
    jones_transmission = np.stack(jt_cols, axis=1)
    return orders2d, R_eff, T_eff, jones_reflection, jones_transmission
