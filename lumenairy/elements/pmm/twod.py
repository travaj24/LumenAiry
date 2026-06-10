"""
lumenairy.elements.pmm.twod -- 2-D crossed-grating Polynomial Modal Method (hybrid).
================================================================================

:func:`pmm_efficiency_2d` is the 2-D (doubly periodic) analogue of
:func:`lumenairy.elements.pmm.pmm_efficiency_1d`, for a **separable rectangular
pillar** (one ``eps_pillar`` rectangle embedded in an ``eps_host`` background).
It is the modal-method counterpart of :func:`lumenairy.elements.rcwa.rcwa_efficiency_2d`.

How it works (and how it differs from the 1-D PMM)
--------------------------------------------------
The 1-D PMM is a *no-floor* subsectional nodal modal method -- its accuracy is
limited only by the polynomial degree, with no Fourier-truncation plateau.  A
fully nodal **2-D** modal method on a tensor-product GLL grid is *not* viable:
the degenerate uniform-region nodal eigenproblem is flux-inconsistent and the
all-nodal solve violates energy conservation (the same wall RCWA hits, which is
why RCWA falls back to its analytic ``W = I`` plane-wave region path).

This solver is therefore a **hybrid**:

* **Structured layer** -- assembled as a tensor-product GLL spectral-element
  nodal operator (so the rectangular pillar is resolved geometrically, not by a
  staircased Fourier series), then **Fourier-Galerkin projected** into the
  Rayleigh (plane-wave) basis.  The projection ``O_F = T O T^+`` gives square
  Fourier-basis layer modes with the correct spectrum and spectral convergence.
* **Half-space regions** (substrate / superstrate) -- treated **analytically**
  as exact plane waves (``W = I``), giving a clean far field and exact flux.

Because the layer is represented in a truncated Rayleigh basis of half-width
``n_orders``, this method **has a Fourier-truncation floor**, exactly like the
FMM/RCWA -- it is *not* no-floor like the 1-D PMM.  Its advantage over RCWA is
that the pillar edge is captured by the nodal grid (geometry-conforming) rather
than by a Gibbs-limited Fourier series, so for a given ``n_orders`` it tends to
be at least as accurate; validated against :func:`rcwa_efficiency_2d` (Li rule,
17 orders) to ``~2e-4`` on the 0-th order at ``degree=11``, energy conserved to
``~1e-3``.

The "null-mode" fix that makes it work
--------------------------------------
A naive nodal ``[Sx; Sy] P@Q`` solve produced a fatal cloud of spurious modes.
The root cause was **not** a fundamental vector-Maxwell spurious-gradient
problem -- it was the classic **periodic-grid Nyquist null mode** of the nodal
first-derivative operator, which appears precisely when the per-axis node count
is *even*.  Forcing the per-axis node count **odd** (``3 * degree *
elements_per_strip`` odd, i.e. odd ``degree``) restores the correct
one-dimensional derivative kernel, and the divergence-reduced second-order
``[Sx; Sy]`` form then injects no spurious modes -- no grad-div penalty,
projection, or Lagrange multiplier needed.

Scope / limitations
-------------------
* :func:`pmm_efficiency_2d` solves a **single separable rectangular pillar**
  (``eps_pillar`` in ``eps_host``); :func:`pmm_efficiency_2d_cell` accepts an
  ARBITRARY AXIS-ALIGNED piecewise-constant ``eps_cell`` grid (any number of
  rectangular regions; walls on the pixel grid are resolved geometrically).
  For curved / non-axis-aligned profiles use :func:`rcwa_efficiency_2d`.
* **Isotropic scalar** TE/TM only in this module's efficiency entry points --
  for anisotropic (3, 3) tensor cells + the full 2x2 Jones use
  :func:`pmm_jones_2d`; for multilayer stacks use :class:`PMM2DStack`.
* **Conical incidence validated to large angles** (v5.14, vs RCWA-2D at theta
  up to 60 deg x arbitrary phi, and vs the 1-D solvers on y-uniform cells):
  the Bloch-shifted operators are correct, with the hybrid's Fourier floor
  growing mildly with ``theta`` at fixed truncation (energy ~2-3e-2 at 60 deg
  with degree=9 / n_orders=4 -- raise ``n_orders`` at steep incidence).
  A WALL-LESS axis is handled analytically (exact ``diag(k)``), so transverse
  momentum on that axis is conserved EXACTLY (a y-uniform cell scatters
  nothing into ``n != 0``; the pre-v5.14 all-nodal path leaked 4-8% there).
  An evanescent-incidence superstrate raises; wavelengths on an EXACT Wood
  anomaly are nudged off it (both adopted from RCWA).
* Single layer of thickness ``depth``; the regions are uniform half-spaces.

CONVENTION (matches the public efficiency convention of ``rcwa_efficiency_2d``):
public ``exp(-i w t)``; ``Gx = (-i/k0) d/dx`` nodal drop-in for ``Kx``;
``lam = sqrt(-kz^2/k0^2)`` on the decay branch; forward ``X = exp(-lam k0 L)``;
**no** eps conjugation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ...backend import is_jax_array
from ..rcwa import Efficiency2D  # cross-suite 2-D result (unpacks (o,R,T), carries .dof)
from ..rcwa._core import (
    _grazing_safe_wavelength,
    _require_propagating_incidence,
)
from ._core import (
    _gll_nodes_weights,
    _graded_boundaries,
    _interface_smatrix,
    _lagrange_derivative_matrix,
    _propagation_smatrix,
    _redheffer_star,
    _stabilize_scalar,
    _StabilizeScanExhausted,
)


def _scan_solver(solve_at, base_degree):
    """Map the stabilize scan index onto consecutive ODD degrees and convert an
    unaffordable higher degree (cost-cap / order-representability ValueError)
    into a graceful end-of-scan."""
    def _at(d):
        deg = base_degree + 2 * (d - base_degree)
        try:
            return solve_at(deg)
        except ValueError:
            if deg > base_degree:
                raise _StabilizeScanExhausted() from None
            raise
    return _at


# 2-D-calibrated stabilize tolerances: the hybrid's Fourier-truncation floor
# (energy ~1e-2, per-order drift between consecutive degrees ~5e-3 at moderate
# settings) sits ABOVE the 1-D no-floor constants, so the 1-D consensus would
# reject every 2-D solve as non-passive.
_PASSIVE_TOL_2D = 5.0e-2
_PER_ORDER_TOL_2D = 1.0e-2

__all__ = ["pmm_efficiency_2d", "pmm_efficiency_2d_cell", "PreparedPMM2D",
           "prepare_pmm_2d", "prepare_pmm_2d_cell",
           "pmm_efficiency_2d_vs_wavelength",
           "pmm_efficiency_2d_cell_vs_wavelength"]

_C = np.complex128


# =========================================================================== #
# 1-D per-axis GLL spectral-element assembly (periodic C0, 3 strips: the pillar
# sits in the middle strip [wall0, wall1]; the two outer strips are host).
# =========================================================================== #

def _build_axis(period, walls, degree, n_el=1, grade=False):
    """Per-axis periodic GLL spectral-element assembly over the strips defined
    by ``walls`` (any number; the original pillar layout passes two).  ``n_el``
    is the element count per strip -- an int (uniform, the historical form) or a
    sequence with one entry per strip (used by the cell path's Nyquist-parity
    bump, which adds one element to the widest strip when the total node count
    would otherwise be even)."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    bnds = [0.0] + list(walls) + [period]
    strips = list(zip(bnds[:-1], bnds[1:]))
    n_el_list = ([int(n_el)] * len(strips) if np.isscalar(n_el)
                 else [int(v) for v in n_el])
    if len(n_el_list) != len(strips):
        raise ValueError(
            f"_build_axis: n_el sequence length {len(n_el_list)} != number of "
            f"strips {len(strips)}")
    elem_bnds = []
    for s, (a, b) in enumerate(strips):
        eb = _graded_boundaries(a, b, n_el_list[s], grade)
        for e in range(len(eb) - 1):
            elem_bnds.append((eb[e], eb[e + 1], s))
    n_el_tot = len(elem_bnds)

    l2g = np.zeros((n_el_tot, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el_tot):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el_tot - 1, degree]      # wrap the last node onto node 0 (periodic)
    l2g[l2g == last] = 0
    n = last

    def _z():
        return np.zeros((n, n), dtype=_C)

    M = _z()
    D = _z()                              # M-weighted (consistent) first derivative
    nstr = len(strips)
    Mtile = [_z() for _ in range(nstr)]
    for e in range(n_el_tot):
        xl, xr, s = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Dloc = Mloc @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        M[ix] += Mloc
        D[ix] += Dloc
        Mtile[s][ix] += Mloc
    return dict(M=M, D=D, Mtile=Mtile, n=n, strips=strips, l2g=l2g,
                elem_bnds=elem_bnds, degree=degree, ref_nodes=ref_nodes,
                period=period)


# =========================================================================== #
# Arbitrary axis-aligned cell geometry: derive EXACT walls + the per-strip eps
# tile from a piecewise-constant pixel grid (the RCWA ``eps_cell`` input form,
# resolved geometrically -- no Fourier staircase).
# =========================================================================== #

def _cell_to_walls_tile(eps_cell, period_x, period_y, fn_name):
    """Derive per-axis wall positions + the ``(nsx, nsy)`` strip tile from an
    axis-aligned ``(Sx, Sy[, ...])`` pixel grid.

    Walls sit on pixel boundaries where any entry of the adjacent pixel
    column/row differs, so the geometry is represented EXACTLY (the pixel grid
    IS the geometry, not a sampling of it) -- every block of the derived
    ``(x-strip, y-strip)`` product grid is provably uniform, so ANY pixel grid
    is exactly representable.  The practical guard is COST, not validity: a
    sampled smooth profile (anti-aliased disk, graded index) makes nearly every
    pixel boundary a wall and explodes the nodal grid -- the entry points cap
    the nodal DOF (see :func:`_validate_cell_cost`) and point to
    :func:`rcwa_efficiency_2d` / :func:`pmm_efficiency_2d_staggered` instead.
    Trailing dimensions (e.g. the ``(3, 3)`` of a tensor cell) ride along into
    the tile untouched.
    """
    cell = np.asarray(eps_cell, dtype=_C)
    if cell.ndim < 2:
        raise ValueError(
            f"{fn_name}: eps_cell must be at least 2-D (Sx, Sy[, ...]), got "
            f"shape {cell.shape}.")
    Sx, Sy = cell.shape[0], cell.shape[1]

    def _walls(S, axis_first, period):
        # boundary i (between pixels i-1 and i) is a wall iff the slices differ
        sl = np.moveaxis(cell, 0 if axis_first else 1, 0)
        diff = [i for i in range(1, S)
                if not np.array_equal(sl[i], sl[i - 1])]
        return [i * period / S for i in diff], diff

    x_walls, ix = _walls(Sx, True, period_x)
    y_walls, iy = _walls(Sy, False, period_y)
    xb = [0] + ix + [Sx]
    yb = [0] + iy + [Sy]
    nsx, nsy = len(xb) - 1, len(yb) - 1
    tile = np.empty((nsx, nsy) + cell.shape[2:], dtype=_C)
    for sx in range(nsx):
        for sy in range(nsy):
            tile[sx, sy] = cell[xb[sx], yb[sy]]
    return x_walls, y_walls, tile


# Default cap on the nodal problem size nodes_x * nodes_y.  With the v5.14
# FACTORIZED assembly (see _scalar_projected_ops) nothing dense of size N x N
# is ever materialized -- the binding costs are the (Nf, N) projector pair
# (~16 * Nf * N bytes each) and the (Tp * vec) @ Tpinv sandwiches
# (O(Nf^2 N) flops) -- so the ceiling is ~40x higher than the old dense-kron
# path's 4000.  N = 150_000 at n_orders = 11 (Nf = 529) is ~2.5 GB of
# projectors; staircased curved cells (a 32-strip-per-axis disk at degree 9 is
# N ~ 83k) now fit.  The pillar default (degree 11, 3 strips) is N = 1089.
_MAX_NODAL_DOF = 150_000


def _validate_pillar_bounds(fn_name, x_bounds, y_bounds, period_x, period_y):
    """Enforce the documented pillar contract ``0 < lo < hi < period`` (v5.14
    robustness audit: INVERTED bounds silently built a negative-width strip and
    returned an energy-conserving but geometrically WRONG answer; degenerate
    bounds crashed with a raw LinAlgError)."""
    for name, (lo, hi), per in (("x_bounds", x_bounds, period_x),
                                ("y_bounds", y_bounds, period_y)):
        lo, hi = float(lo), float(hi)
        if not (0.0 < lo < hi < float(per)):
            raise ValueError(
                f"{fn_name}: {name} must satisfy 0 < lo < hi < period "
                f"(strictly interior, non-degenerate walls); got "
                f"({lo:.6g}, {hi:.6g}) with period {float(per):.6g}.  A pillar "
                f"touching the cell edge can be expressed by shifting the "
                f"unit-cell origin (the lattice is periodic).")


def _validate_cell_cost(fn_name, el_x, el_y, degree, max_nodal_dof):
    nx = sum(el_x) * degree
    ny = sum(el_y) * degree
    if nx * ny > max_nodal_dof:
        raise ValueError(
            f"{fn_name}: the derived strip layout needs a {nx} x {ny} nodal "
            f"grid ({nx * ny} nodal DOF > max_nodal_dof={max_nodal_dof}); the "
            f"(n_orders-sized x N) projector pair + sandwiches would be too "
            f"large.  A very large strip count usually means eps_cell is a "
            f"SAMPLED SMOOTH profile (anti-aliased graded index) with "
            f"({len(el_x)} x-strips, {len(el_y)} y-strips).  STAIRCASED curved "
            f"cells are supported up to the cap (v5.14 factorized assembly); "
            f"beyond it use rcwa_efficiency_2d (Fourier sampling) or "
            f"pmm_efficiency_2d_staggered, or lower degree / "
            f"elements_per_strip, or raise max_nodal_dof explicitly.")


def _axis_elem_counts(period, walls, degree, elements_per_strip, fn_name,
                      axis_name):
    """Per-strip element counts for an N-strip axis, with the periodic-Nyquist
    parity bump: the per-axis node count ``n_el_tot * degree`` must be ODD (an
    even count puts an exact Nyquist null in the periodic first-derivative
    kernel -- the module-docstring failure mode).  ``degree`` must be odd; when
    the strip count makes ``n_el_tot`` even, the WIDEST strip gets one extra
    element (a pure hp-refinement -- the geometry and the converged answer are
    unchanged)."""
    if degree % 2 == 0:
        raise ValueError(
            f"{fn_name}: degree must be ODD (per-axis node count n_el*degree "
            f"must be odd to avoid the periodic Nyquist null mode); got "
            f"degree={degree}")
    bnds = [0.0] + list(walls) + [period]
    n_strips = len(bnds) - 1
    counts = [int(elements_per_strip)] * n_strips
    if (sum(counts) * degree) % 2 == 0:
        widths = [b - a for a, b in zip(bnds[:-1], bnds[1:])]
        counts[int(np.argmax(widths))] += 1
    return counts


# =========================================================================== #
# 2-D Kronecker assembly of the RCWA-normalized operators.
#   vec ordering x-inner: index I = jx + nx*jy  ->  kron(Y, X)
# =========================================================================== #

def _assemble_2d(ax, ay, eps_tile, k0):
    Mx, Dx = ax["M"], ax["D"]
    My, Dy = ay["M"], ay["D"]
    nx, ny = ax["n"], ay["n"]
    N = nx * ny
    M = np.kron(My, Mx)
    DX = np.kron(My, Dx)
    DY = np.kron(Dy, Mx)
    Minv = np.linalg.inv(M)
    Gx = (-1j / k0) * (Minv @ DX)
    Gy = (-1j / k0) * (Minv @ DY)

    nsx, nsy = len(ax["Mtile"]), len(ay["Mtile"])
    P_eps = np.zeros((N, N), dtype=_C)
    P_inv = np.zeros((N, N), dtype=_C)
    for sx in range(nsx):
        for sy in range(nsy):
            e = eps_tile[sx, sy]
            ker = np.kron(ay["Mtile"][sy], ax["Mtile"][sx])
            P_eps += e * ker
            P_inv += (1.0 / e) * ker
    Eps = Minv @ P_eps               # multiply-by-eps  (Laurent / tangential)
    Einv = Minv @ P_inv              # multiply-by-1/eps (E_z elimination, Li)
    Epn = np.linalg.solve(P_inv, M)  # inverse rule [[1/eps]]^-1 (wall-normal Ex)
    return dict(Gx=Gx, Gy=Gy, Eps=Eps, Einv=Einv, Epn=Epn, M=M, N=N)


# =========================================================================== #
# decay-branch helpers + plane-wave kz
# =========================================================================== #

def _sqrt_decay(x):
    r = np.sqrt(np.asarray(x, dtype=_C))
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


def _inv_lam(lam):
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _kz_forward2(eps, kx, ky):
    """``kz/k0`` on the forward branch for ``exp(-i w t)`` (2-D)."""
    val = np.sqrt(np.asarray(eps - kx ** 2 - ky ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


# =========================================================================== #
# Far-field nodal -> Rayleigh projection  T[m,i] = (1/L) INT phi_i e^{-imGx}
# =========================================================================== #

def _axis_projection(ax, orders_1d):
    from numpy.polynomial.legendre import leggauss
    l2g, elem_bnds, degree = ax["l2g"], ax["elem_bnds"], ax["degree"]
    ref_nodes, n, L = ax["ref_nodes"], ax["n"], ax["period"]
    G = 2.0 * np.pi / L
    nq = max(2 * degree + 8, 24)
    xg, wg = leggauss(nq)
    wbary = np.ones(degree + 1)
    for j in range(degree + 1):
        for k in range(degree + 1):
            if k != j:
                wbary[j] /= (ref_nodes[j] - ref_nodes[k])

    def _vals(xi):
        V = np.zeros((len(xi), degree + 1))
        for r, x in enumerate(xi):
            diff = x - ref_nodes
            if np.any(np.abs(diff) < 1e-14):
                V[r, np.argmin(np.abs(diff))] = 1.0
            else:
                num = wbary / diff
                V[r, :] = num / num.sum()
        return V

    Lv = _vals(xg)
    T = np.zeros((len(orders_1d), n), dtype=_C)
    for e in range(len(elem_bnds)):
        xl, xr, _s = elem_bnds[e]
        J = 0.5 * (xr - xl)
        xphys = 0.5 * (xr + xl) + J * xg
        phase = np.exp(-1j * np.outer(np.asarray(orders_1d) * G, xphys))
        contrib = (phase * (wg * J / L)) @ Lv
        idx = l2g[e]
        for a in range(degree + 1):
            T[:, idx[a]] += contrib[:, a]
    return T


def _projectors(ax, ay, ox, oy):
    Tx = _axis_projection(ax, ox)
    Ty = _axis_projection(ay, oy)
    Tp = np.kron(Ty, Tx)
    Tpinv = np.linalg.pinv(Tp)
    return Tp, Tpinv


# =========================================================================== #
# layer modes (projected nodal) + analytic region modes
# =========================================================================== #

def _axis_ops_1d(axd, eps_1d):
    """1-D nodal operators for one axis (k0-FREE): the unit derivative
    ``G0 = -i M^-1 D`` plus the multiply-by-eps / 1/eps / inverse-rule
    operators for the per-strip profile ``eps_1d``."""
    M, D = axd["M"], axd["D"]
    Minv = np.linalg.inv(M)
    G0 = -1j * (Minv @ D)
    P_eps = np.zeros_like(M)
    P_inv = np.zeros_like(M)
    for s, Mt in enumerate(axd["Mtile"]):
        P_eps += eps_1d[s] * Mt
        P_inv += (1.0 / eps_1d[s]) * Mt
    return dict(G0=G0, Eps=Minv @ P_eps, Einv=Minv @ P_inv,
                Epn=np.linalg.solve(P_inv, M))


def _scalar_projected_ops(ax, ay, eps_tile, ox, oy, period_x, period_y):
    """K0-FREE Fourier-projected scalar layer operators, with per-axis EXACT
    handling of WALL-LESS axes.

    When an axis has no walls the cell is uniform along it, so the 2-D
    operators kron-FACTOR: that axis's derivative is exactly ``diag(k)`` in
    the Fourier basis and its projection is the identity.  Building those
    axes analytically (instead of projecting a 1-strip nodal grid) makes
    transverse-momentum conservation EXACT -- a y-uniform cell scatters NOTHING
    into ``n != 0`` orders, restoring the 1-D reduction that the all-nodal path
    violated at oblique incidence (the projected nodal derivative of a
    coarse wall-less axis aliases) -- and is much cheaper.

    Returns a dict of ``Gx0F/Gy0F`` (unit derivative operators: the caller
    forms ``GxF = Gx0F/k0 + kx0*IpxF``), ``EpsF/EinvF/EpnF``, and the per-axis
    projected identities ``IpxF/IpyF``.
    """
    NxO, NyO = len(ox), len(oy)
    nsx = len(ax["strips"])
    nsy = len(ay["strips"])
    if nsx > 1 and nsy > 1:
        # FACTORIZED assembly (v5.14 perf audit P1): the GLL masses are exactly
        # DIAGONAL, so Minv @ DX = kron(I, Mx^-1 Dx), every eps operator is a
        # diagonal NODAL VECTOR, and pinv(kron(Ty, Tx)) = kron(pinv, pinv) --
        # the dense N x N kron materialization + the O(N^3) LAPACK inversion
        # OF A DIAGONAL MATRIX the old _assemble_2d path performed are never
        # needed.  Measured machine-identical to the dense path (rel ~2.6e-15)
        # and 220-1078x faster / 64-138x less memory at degree 9-11; the dense
        # assembly was 88-97% of the whole solve.
        Tx = _axis_projection(ax, ox)
        Txp = np.linalg.pinv(Tx)
        Ty = _axis_projection(ay, oy)
        Typ = np.linalg.pinv(Ty)
        mdx = np.diag(ax["M"])
        mdy = np.diag(ay["M"])
        gx1 = Tx @ ((1.0 / mdx)[:, None] * ax["D"]) @ Txp
        gy1 = Ty @ ((1.0 / mdy)[:, None] * ay["D"]) @ Typ
        Gx0F = -1j * np.kron(np.eye(NyO, dtype=_C), gx1)
        Gy0F = -1j * np.kron(gy1, np.eye(NxO, dtype=_C))
        Mdiag = np.kron(mdy, mdx)
        P_eps = np.zeros_like(Mdiag)
        P_inv = np.zeros_like(Mdiag)
        for sx in range(nsx):
            for sy in range(nsy):
                ker = np.kron(np.diag(ay["Mtile"][sy]),
                              np.diag(ax["Mtile"][sx]))
                e = eps_tile[sx, sy]
                P_eps += e * ker
                P_inv += ker / e
        eps_nodal = P_eps / Mdiag
        inv_nodal = P_inv / Mdiag
        Tp = np.kron(Ty, Tx)
        Tpinv = np.kron(Typ, Txp)
        EpsF = (Tp * eps_nodal[None, :]) @ Tpinv
        EinvF = (Tp * inv_nodal[None, :]) @ Tpinv
        EpnF = (Tp * (1.0 / inv_nodal)[None, :]) @ Tpinv
        Ip = np.kron(Ty @ Typ, Tx @ Txp)
        return dict(Gx0F=Gx0F, Gy0F=Gy0F, EpsF=EpsF, EinvF=EinvF, EpnF=EpnF,
                    IpxF=Ip, IpyF=Ip)
    if nsy == 1 and nsx > 1:                    # uniform along y
        o1 = _axis_ops_1d(ax, eps_tile[:, 0])
        Tx = _axis_projection(ax, ox)
        Txp = np.linalg.pinv(Tx)
        Iy = np.eye(NyO, dtype=_C)
        ipx = Tx @ Txp
        Gy0F = np.diag(np.repeat(oy, NxO).astype(_C)
                       * (2.0 * np.pi / period_y))
        return dict(Gx0F=np.kron(Iy, Tx @ o1["G0"] @ Txp), Gy0F=Gy0F,
                    EpsF=np.kron(Iy, Tx @ o1["Eps"] @ Txp),
                    EinvF=np.kron(Iy, Tx @ o1["Einv"] @ Txp),
                    EpnF=np.kron(Iy, Tx @ o1["Epn"] @ Txp),
                    IpxF=np.kron(Iy, ipx),
                    IpyF=np.eye(NxO * NyO, dtype=_C))
    if nsx == 1 and nsy > 1:                    # uniform along x
        o1 = _axis_ops_1d(ay, eps_tile[0, :])
        Ty = _axis_projection(ay, oy)
        Typ = np.linalg.pinv(Ty)
        Ix = np.eye(NxO, dtype=_C)
        ipy = Ty @ Typ
        Gx0F = np.diag(np.tile(ox, NyO).astype(_C)
                       * (2.0 * np.pi / period_x))
        return dict(Gx0F=Gx0F, Gy0F=np.kron(Ty @ o1["G0"] @ Typ, Ix),
                    EpsF=np.kron(Ty @ o1["Eps"] @ Typ, Ix),
                    EinvF=np.kron(Ty @ o1["Einv"] @ Typ, Ix),
                    EpnF=np.kron(Ty @ o1["Epn"] @ Typ, Ix),
                    IpxF=np.eye(NxO * NyO, dtype=_C),
                    IpyF=np.kron(ipy, Ix))
    # no walls on either axis: a uniform cell (the callers shortcut this to
    # analytic plane-wave modes before reaching here) -- exact diagonals.
    eps0 = eps_tile.flat[0]
    NfI = np.eye(NxO * NyO, dtype=_C)
    Gx0F = np.diag(np.tile(ox, NyO).astype(_C) * (2.0 * np.pi / period_x))
    Gy0F = np.diag(np.repeat(oy, NxO).astype(_C) * (2.0 * np.pi / period_y))
    return dict(Gx0F=Gx0F, Gy0F=Gy0F, EpsF=eps0 * NfI, EinvF=NfI / eps0,
                EpnF=eps0 * NfI, IpxF=NfI, IpyF=NfI)


def _layer_modes_projected(GxF, GyF, EpsF, EinvF, EpnF, formulation="li"):
    Nf = GxF.shape[0]
    I = np.eye(Nf, dtype=_C)
    EPS_normal = EpnF if formulation == "li" else EpsF
    Q = np.block([[GxF @ GyF, EpsF - GxF @ GxF],
                  [GyF @ GyF - EPS_normal, -GyF @ GxF]])
    EPS_inv = EinvF if formulation == "li" else np.linalg.inv(EpsF)
    P = np.block([[GxF @ EPS_inv @ GyF, I - GxF @ EPS_inv @ GxF],
                  [GyF @ EPS_inv @ GyF - I, -GyF @ EPS_inv @ GxF]])
    lam2, W = np.linalg.eig(P @ Q)
    lam = _sqrt_decay(lam2)
    V = Q @ W @ np.diag(_inv_lam(lam))
    return W, V, lam


def _homogeneous_modes(kx, ky, eps):
    """Analytic plane-wave (Rayleigh) modes: W=I, V=Q diag(1/lam)."""
    N = len(kx)
    Kx = np.diag(kx.astype(_C))
    Ky = np.diag(ky.astype(_C))
    kz = _kz_forward2(eps, kx, ky)
    lam = _sqrt_decay(-np.concatenate([kz, kz]) ** 2)
    eps_I = eps * np.eye(N, dtype=_C)
    Q = np.block([[Kx @ Ky, eps_I - Kx @ Kx],
                  [Ky @ Ky - eps_I, -Ky @ Kx]])
    W = np.eye(2 * N, dtype=_C)
    V = Q @ np.diag(_inv_lam(lam))
    return W, V, lam, kz


# =========================================================================== #
# Shared scalar solve core (geometry-agnostic: any N-strip axis layout).
# All eps arguments arrive in the INTERNAL (conjugated, exp(+iwt)) convention;
# walls / per-strip element counts are already validated by the entry points.
# =========================================================================== #

def _pmm2d_solve_core(period_x, period_y, x_walls, y_walls, eps_tile,
                      eps_sup, eps_sub, depth, wavelength, degree, el_x, el_y,
                      grade, polarization, theta, phi, n_orders, formulation,
                      fn_name="pmm_efficiency_2d"):
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

    # Conical-incidence hardening (mirrors rcwa_efficiency_2d): reject an
    # evanescent incident wave (grazing/metallic superstrate -> kz_inc ~ 0
    # divides the flux normalization), and nudge the wavelength off any EXACT
    # Wood anomaly in any constituent medium (a grazing order crashes the
    # interface S-matrix).  The nudge is identity away from exact grazing, so
    # ordinary solves are byte-unchanged.  Re() is conjugation-invariant, so
    # the internal-convention eps values are safe to pass.
    _require_propagating_incidence(fn_name, eps_sup, kx0 ** 2 + ky0 ** 2)
    eps_reals = [eps_sup, eps_sub] + [complex(e) for e in
                                      np.asarray(eps_tile).ravel()]
    wl = _grazing_safe_wavelength(float(wavelength), kx0, ky0, order_x,
                                  order_y, period_x, period_y, eps_reals)
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)

    # ---- analytic Fourier regions (exact plane waves: flux-exact far field) --
    Wsup, Vsup, _ls, kz_ref = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, kz_trn = _homogeneous_modes(kxv, kyv, eps_sub)

    # ---- layer modes: projected nodal, UNLESS laterally uniform (-> analytic)
    eps0 = eps_tile.flat[0]
    uniform_layer = bool(np.all(np.abs(eps_tile - eps0) < 1e-12))
    if uniform_layer:
        Wl, Vl, lam_l, _ = _homogeneous_modes(kxv, kyv, eps0)
    else:
        lops = _scalar_projected_ops(ax, ay, eps_tile, ox, oy, period_x,
                                     period_y)
        GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
        GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
        Wl, Vl, lam_l = _layer_modes_projected(
            GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
            formulation=formulation)

    # ---- Redheffer recursion (Fourier/Rayleigh basis) ----
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # ---- incident (0,0) plane wave (Rayleigh basis: amplitudes ARE coeffs) --
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
            kz_inc0 = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    cinc = np.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:Nf], r[Nf:]
    tx, ty = t[:Nf], t[Nf:]

    kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
    # PUBLIC-convention forward kz for the z-flux + mask + Ez (the hybrid PMM
    # conjugates the region eps for its internal solve; un-conjugate here so a
    # lossy exit substrate is not read as evanescent -> T zeroed; lossless eps is
    # real so this is byte-unchanged).  Mirrors rcwa._core._forward_flux_kz.
    kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
    safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R = np.real(kz_ref_f / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                      + np.abs(rz) ** 2) / einc_sq
    T = np.real(kz_trn_f / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                      + np.abs(tz) ** 2) / einc_sq
    R = np.where(np.real(kz_ref_f) > 0, np.real(R), 0.0)
    T = np.where(np.real(kz_trn_f) > 0, np.real(T), 0.0)
    orders2d = np.stack([order_x, order_y], axis=1)
    # cross-suite return shape: unpacks as (orders, R, T); .dof = 2*Nf (the modal
    # eigenproblem dimension).  Was a bare 4-tuple (orders, R, T, dof) pre-v5.12.
    return Efficiency2D(orders2d, R, T, 2 * Nf)


# =========================================================================== #
# public driver
# =========================================================================== #

def pmm_efficiency_2d(
    period_x: float,
    period_y: float,
    eps_pillar: complex,
    eps_host: complex,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
    stabilize: bool = False,
) -> Efficiency2D:
    r"""Diffraction efficiencies of a 2-D rectangular pillar via the hybrid PMM.

    The modal-method counterpart of :func:`rcwa_efficiency_2d` for a single
    separable rectangular pillar (``eps_pillar`` rectangle in an ``eps_host``
    background).  See the module docstring for the method, the null-mode fix,
    and -- importantly -- the **scope limitations** (single rect pillar,
    isotropic scalar TE/TM, normal/near-normal incidence, single layer; this is
    a hybrid with a Fourier-truncation ``n_orders`` floor, *not* no-floor like
    the 1-D PMM).

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_pillar, eps_host : complex
        Relative permittivities of the pillar and the surrounding host.
    x_bounds, y_bounds : (float, float)
        ``(x0, x1)`` and ``(y0, y1)`` rectangle edges (metres), ``0 <= x0 <
        x1 <= period_x`` (and likewise in y).
    n_substrate, n_superstrate : complex
        Refractive indices of the transmission (substrate) and incidence
        (superstrate) half-spaces.  Internally squared to permittivities.
    depth : float
        Pillar height / layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    degree : int, optional
        GLL polynomial degree per element.  With 3 strips x
        ``elements_per_strip`` elements the per-axis node count is
        ``3 * degree * elements_per_strip``, which **must be odd** (avoids the
        periodic Nyquist null mode); an odd ``degree`` with the defaults
        satisfies this.
    elements_per_strip : int, optional
        Spectral elements per strip (per axis).  Default 1.
    grade : bool, optional
        Graded element boundaries within each strip (clustering toward walls).
    polarization : {"te", "tm"}, optional
        Incident linear polarization.
    theta, phi : float, optional
        Incidence polar / azimuth angles (radians).  Validated near normal;
        large ``theta`` is experimental.
    n_orders : int, optional
        Fourier (Rayleigh) truncation half-width for both regions and the
        projected layer.  Must satisfy ``2*n_orders + 1 <= 3 * degree *
        elements_per_strip`` so the nodal grid can represent the harmonics.
    formulation : {"li", "laurent"}, optional
        Factorization rule for the layer operator (``"li"`` = inverse rule for
        the wall-normal field component; recommended).

    Returns
    -------
    result : :class:`~lumenairy.elements.rcwa.Efficiency2D`
        A ``tuple`` subclass that unpacks as ``orders, R, T`` (NOT a 4-tuple) and
        carries the modal degrees of freedom on the ``.dof`` attribute:

        * ``orders`` -- ``(M, 2)`` ``(m, n)`` diffraction-order indices;
        * ``R, T`` -- ``(M,)`` reflected / transmitted efficiencies per order;
        * ``result.dof`` -- modal DOF retained (``2 * (2*n_orders+1)**2``).

        .. note:: **API change (v5.11 -> v5.12).**  This used to return a bare
           4-tuple ``(orders, R, T, dof)``; it now returns the cross-suite
           :class:`Efficiency2D` (shared with :func:`rcwa_efficiency_2d`), which
           unpacks to ``orders, R, T`` with ``dof`` as an attribute.  Migrate
           ``o, R, T, dof = pmm_efficiency_2d(...)`` to either
           ``res = pmm_efficiency_2d(...); o, R, T = res; dof = res.dof`` or
           ``o, R, T = pmm_efficiency_2d(...)[:3]``.
    """
    n_nodes_axis = 3 * degree * elements_per_strip
    if n_nodes_axis % 2 == 0:
        raise ValueError(
            "per-axis node count 3*degree*elements_per_strip must be ODD to "
            "avoid the periodic Nyquist null mode (use an odd polynomial "
            f"degree); got degree={degree}, elements_per_strip="
            f"{elements_per_strip} -> {n_nodes_axis} nodes")
    if 2 * n_orders + 1 > n_nodes_axis:
        raise ValueError(
            f"n_orders={n_orders} too large for degree={degree}: need "
            f"2*n_orders+1 ({2 * n_orders + 1}) <= per-axis nodes "
            f"({n_nodes_axis}); raise degree or lower n_orders")
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")

    # JAX auto-dispatch (Phase 7): traced material values / depth / wavelength
    # / angles on STATIC pillar bounds + degree + orders -- the 2-D analogue of
    # the 1-D binary twin's scope.  See pmm/_jax_twod.py for the caveats.
    _jx = (eps_pillar, eps_host, n_substrate, n_superstrate, depth,
           wavelength, theta, phi)
    if any(is_jax_array(a) for a in _jx):
        if stabilize:
            raise ValueError(
                "pmm_efficiency_2d: stabilize=True is not differentiable "
                "(host-side degree-scan consensus); pass stabilize=False on "
                "the JAX path.")
        from ._jax_twod import _pmm_efficiency_2d_jax
        o, R, T = _pmm_efficiency_2d_jax(
            period_x, period_y, eps_pillar, eps_host, x_bounds, y_bounds,
            n_substrate, n_superstrate, depth, wavelength, degree=degree,
            elements_per_strip=elements_per_strip, grade=grade,
            polarization=polarization, theta=theta, phi=phi,
            n_orders=n_orders, formulation=formulation)
        return Efficiency2D(o, R, T, 2 * (2 * n_orders + 1) ** 2)

    _validate_pillar_bounds("pmm_efficiency_2d", x_bounds, y_bounds,
                            period_x, period_y)
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    # Loss-convention bridge (matches pmm_efficiency_1d / rcwa_efficiency_2d):
    # conjugate the PUBLIC eps (Im>0 for loss) into the internal exp(+iwt)
    # convention the modal solve + decay branch use; real (lossless) eps is
    # unaffected (conj is identity there), so the validated dielectric path is
    # byte-unchanged.  Without this a passive lossy material is read as GAIN
    # (R+T>1).  The efficiencies are real flux ratios, so no output un-conjugation
    # is needed.
    eps_p = np.conj(_C(eps_pillar))
    eps_h = np.conj(_C(eps_host))
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    eps_tile = np.full((3, 3), eps_h, dtype=_C)
    eps_tile[1, 1] = eps_p

    def _solve_at(deg):
        # cost cap inside the scan closure too (v5.14 robustness audit: the
        # stabilize degree ladder on a config that is non-passive at EVERY
        # degree previously ran away to multi-GB dense problems)
        nodes = 3 * elements_per_strip * deg
        if nodes * nodes > _MAX_NODAL_DOF:
            raise ValueError(
                f"pmm_efficiency_2d: degree {deg} needs a {nodes} x {nodes} "
                f"nodal grid ({nodes * nodes} > max_nodal_dof "
                f"{_MAX_NODAL_DOF}).")
        return _pmm2d_solve_core(
            period_x, period_y, [x0, x1], [y0, y1], eps_tile, eps_sup,
            eps_sub, depth, wavelength, deg, elements_per_strip,
            elements_per_strip, grade, polarization, theta, phi, n_orders,
            formulation)

    if not stabilize:
        return _solve_at(degree)
    o, R, T = _stabilize_scalar(_scan_solver(_solve_at, degree), degree,
                                "pmm_efficiency_2d",
                                passive_tol=_PASSIVE_TOL_2D,
                                per_order_tol=_PER_ORDER_TOL_2D)
    return Efficiency2D(o, R, T, 2 * (2 * n_orders + 1) ** 2)


def _validate_cell_orders(fn_name, n_orders, degree, el_x, el_y):
    """Per-axis Rayleigh-representability check for the cell path (the cell's
    strip count sets the node count per axis, unlike the fixed-3-strip pillar).
    A WALL-LESS axis (1 strip) is exempt: it is handled analytically in the
    Fourier basis (exact ``diag(k)``; see :func:`_scalar_projected_ops`), so it
    needs no nodal resolution at all."""
    for counts, name in ((el_x, "x"), (el_y, "y")):
        if len(counts) == 1:
            continue                       # wall-less axis: exact, no node need
        nodes = sum(counts) * degree
        if 2 * n_orders + 1 > nodes:
            raise ValueError(
                f"{fn_name}: n_orders={n_orders} too large for the {name}-axis "
                f"grid: need 2*n_orders+1 ({2 * n_orders + 1}) <= per-axis "
                f"nodes ({nodes} = {sum(counts)} elements x degree {degree}); "
                f"raise degree / elements_per_strip or lower n_orders")


def pmm_efficiency_2d_cell(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
    max_nodal_dof: int = _MAX_NODAL_DOF,
    stabilize: bool = False,
) -> Efficiency2D:
    """Diffraction efficiencies of an ARBITRARY axis-aligned piecewise-constant
    2-D cell via the hybrid PMM -- the multi-region generalization of
    :func:`pmm_efficiency_2d` (any number of rectangular regions: multiple
    pillars, crosses, interdigitated pads, ...).

    The ``(Sx, Sy)`` ``eps_cell`` grid IS the geometry: every pixel boundary
    where the adjacent column/row differs becomes an exact spectral-element
    wall, so the regions are resolved geometrically (no Fourier staircase) --
    the input convention of :func:`rcwa_efficiency_2d`, resolved exactly.
    Intended for cells made of a MODEST number of axis-aligned rectangles; a
    sampled smooth profile (anti-aliased disk, graded index) turns nearly every
    pixel boundary into a wall and trips the ``max_nodal_dof`` cost guard (use
    :func:`rcwa_efficiency_2d` or :func:`pmm_efficiency_2d_staggered` there).

    Parameters are otherwise as in :func:`pmm_efficiency_2d`.  The per-axis node
    count is ``n_strips * elements_per_strip * degree`` (geometry-dependent);
    when that total is even, one extra element is added to the widest strip on
    that axis (a pure hp-refinement) to keep the periodic node count odd (the
    Nyquist-null fix -- see the module docstring).  ``max_nodal_dof`` caps the
    dense nodal problem size ``nodes_x * nodes_y`` (default 4000, ~256 MB per
    dense operator).  ``stabilize=True`` runs the 1-D PMM's per-order
    degree-scan consensus (guards the measure-zero quasi-resonances; each scan
    step re-solves at the next ODD degree -- expensive in 2-D, so the default
    is False).

    Returns
    -------
    result : :class:`~lumenairy.elements.rcwa.Efficiency2D`
        Unpacks as ``orders, R, T``; ``.dof = 2 * (2*n_orders+1)**2``.
    """
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    x_walls, y_walls, tile = _cell_to_walls_tile(
        eps_cell, period_x, period_y, "pmm_efficiency_2d_cell")
    # Loss-convention bridge (see pmm_efficiency_2d).
    eps_tile = np.conj(tile)
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    def _solve_at(deg):
        el_x = _axis_elem_counts(period_x, x_walls, deg, elements_per_strip,
                                 "pmm_efficiency_2d_cell", "x")
        el_y = _axis_elem_counts(period_y, y_walls, deg, elements_per_strip,
                                 "pmm_efficiency_2d_cell", "y")
        _validate_cell_orders("pmm_efficiency_2d_cell", n_orders, deg, el_x,
                              el_y)
        _validate_cell_cost("pmm_efficiency_2d_cell", el_x, el_y, deg,
                            max_nodal_dof)
        return _pmm2d_solve_core(
            period_x, period_y, x_walls, y_walls, eps_tile, eps_sup, eps_sub,
            depth, wavelength, deg, el_x, el_y, grade, polarization, theta,
            phi, n_orders, formulation, fn_name="pmm_efficiency_2d_cell")

    if not stabilize:
        return _solve_at(degree)
    # degree-scan consensus on consecutive ODD degrees (the scan index d is
    # mapped 1:1 onto degree, degree+2, ...; an unaffordable higher degree
    # ends the scan gracefully)
    o, R, T = _stabilize_scalar(_scan_solver(_solve_at, degree), degree,
                                "pmm_efficiency_2d_cell",
                                passive_tol=_PASSIVE_TOL_2D,
                                per_order_tol=_PER_ORDER_TOL_2D)
    return Efficiency2D(o, R, T, 2 * (2 * n_orders + 1) ** 2)


# =========================================================================== #
# Wavelength-sweep reuse: assemble the GEOMETRY-ONLY work once, sweep many.
#
# This is the HIGHEST-LEVERAGE sweep in the library: the expensive parts of the
# hybrid PMM-2D -- the nodal mass inverse + the [[1/eps]]^-1 nodal inversion
# (``Eps``/``Einv``/``Epn``) and the ``O(N^3)`` Fourier-projection pseudo-inverse
# (``Tp``/``Tpinv``) -- are all ``k0``-FREE and depend only on the basis +
# geometry + order set.  Their Fourier-projected forms ``EpsF``/``EinvF``/``EpnF``
# and ``Gx0F``/``Gy0F`` (the projected derivative operators at ``k0 = 1``) are
# likewise wavelength-independent.  Per wavelength only the cheap rescale
# ``GxF = Gx0F/k0 + kx0*IprojF`` and the SMALL projected eig (size ``2*Nf``,
# ``Nf = (2 n_orders + 1)^2``) + S-matrix cascade run.  When ``degree >>
# n_orders`` the reusable nodal/pinv work dominates -> a multi-x sweep speed-up.
# ``prepare_pmm_2d(...).solve(wl)`` reproduces ``pmm_efficiency_2d(...)`` to
# ~1e-13 (the only delta: ``Gx0F/k0`` reorders one division vs the single call;
# the uniform-layer path is byte-identical).  Non-dispersive + fixed (theta,phi).
# =========================================================================== #
class PreparedPMM2D:
    """A wavelength-invariant hybrid 2-D PMM assembly, reusable across a sweep.

    Build with :func:`prepare_pmm_2d`; call :meth:`solve` per wavelength.  Holds
    the basis, the order set, the incident vector, and -- for a structured layer
    -- the ``k0``-free Fourier-projected operators (the reused nodal inversions +
    pseudo-inverse).  :meth:`solve` runs only the per-wavelength operator rescale,
    the small projected eig, and the S-matrix cascade.  NON-DISPERSIVE indices and
    a FIXED ``(theta, phi)`` are assumed.
    """

    __slots__ = ("period_x", "period_y", "depth", "polarization", "formulation",
                 "eps_h", "eps_sup", "eps_sub", "uniform_layer",
                 "order_x", "order_y", "Nf", "kx0", "ky0",
                 "Gx0F", "Gy0F", "EpsF", "EinvF", "EpnF", "IpxF", "IpyF",
                 "cinc", "einc_sq", "kz_inc", "eps_reals")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def solve(self, wavelength) -> Efficiency2D:
        """Diffraction efficiencies at ``wavelength`` reusing the prepared
        geometry (equivalent to ``pmm_efficiency_2d(...)`` to ~1e-13)."""
        order_x, order_y, Nf = self.order_x, self.order_y, self.Nf
        wl = _grazing_safe_wavelength(
            float(wavelength), self.kx0, self.ky0, order_x, order_y,
            self.period_x, self.period_y, self.eps_reals)
        k0 = 2.0 * np.pi / wl
        kxv = self.kx0 + order_x * (wl / self.period_x)
        kyv = self.ky0 + order_y * (wl / self.period_y)

        Wsup, Vsup, _ls, kz_ref = _homogeneous_modes(kxv, kyv, self.eps_sup)
        Wsub, Vsub, _lb, kz_trn = _homogeneous_modes(kxv, kyv, self.eps_sub)

        if self.uniform_layer:
            Wl, Vl, lam_l, _ = _homogeneous_modes(kxv, kyv, self.eps_h)
        else:
            GxF = self.Gx0F / k0 + self.kx0 * self.IpxF
            GyF = self.Gy0F / k0 + self.ky0 * self.IpyF
            Wl, Vl, lam_l = _layer_modes_projected(
                GxF, GyF, self.EpsF, self.EinvF, self.EpnF,
                formulation=self.formulation)

        S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
        S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * self.depth))
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
        S11, _S12, S21, _S22 = S

        cinc = self.cinc
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:Nf], r[Nf:]
        tx, ty = t[:Nf], t[Nf:]
        kz_ref_f = _kz_forward2(np.conj(self.eps_sup), kxv, kyv)
        kz_trn_f = _kz_forward2(np.conj(self.eps_sub), kxv, kyv)
        safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
        safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        R = np.real(kz_ref_f / self.kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                               + np.abs(rz) ** 2) / self.einc_sq
        T = np.real(kz_trn_f / self.kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                               + np.abs(tz) ** 2) / self.einc_sq
        R = np.where(np.real(kz_ref_f) > 0, np.real(R), 0.0)
        T = np.where(np.real(kz_trn_f) > 0, np.real(T), 0.0)
        orders2d = np.stack([order_x, order_y], axis=1)
        return Efficiency2D(orders2d, R, T, 2 * Nf)


def prepare_pmm_2d(
    period_x: float,
    period_y: float,
    eps_pillar: complex,
    eps_host: complex,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
) -> PreparedPMM2D:
    """Assemble the wavelength-INDEPENDENT part of a hybrid 2-D PMM solve once,
    for a wavelength sweep that reuses it (see :class:`PreparedPMM2D`).

    Parameters are the geometry/angle subset of :func:`pmm_efficiency_2d` (no
    ``wavelength``).  NON-DISPERSIVE indices and a FIXED ``(theta, phi)`` assumed.
    """
    n_nodes_axis = 3 * degree * elements_per_strip
    if n_nodes_axis % 2 == 0:
        raise ValueError(
            "per-axis node count 3*degree*elements_per_strip must be ODD to "
            "avoid the periodic Nyquist null mode (use an odd polynomial "
            f"degree); got degree={degree}, elements_per_strip="
            f"{elements_per_strip} -> {n_nodes_axis} nodes")
    if 2 * n_orders + 1 > n_nodes_axis:
        raise ValueError(
            f"n_orders={n_orders} too large for degree={degree}: need "
            f"2*n_orders+1 ({2 * n_orders + 1}) <= per-axis nodes "
            f"({n_nodes_axis}); raise degree or lower n_orders")
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")

    _validate_pillar_bounds("prepare_pmm_2d", x_bounds, y_bounds,
                            period_x, period_y)
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    eps_p = np.conj(_C(eps_pillar))
    eps_h = np.conj(_C(eps_host))
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    eps_tile = np.full((3, 3), eps_h, dtype=_C)
    eps_tile[1, 1] = eps_p
    return _prepare_pmm2d_core(
        period_x, period_y, [x0, x1], [y0, y1], eps_tile, eps_sup, eps_sub,
        depth, degree, elements_per_strip, elements_per_strip, grade,
        polarization, theta, phi, n_orders, formulation)


def _prepare_pmm2d_core(period_x, period_y, x_walls, y_walls, eps_tile,
                        eps_sup, eps_sub, depth, degree, el_x, el_y, grade,
                        polarization, theta, phi, n_orders, formulation):
    """Shared geometry-only assembly behind :func:`prepare_pmm_2d` /
    :func:`prepare_pmm_2d_cell` (eps already in the internal convention)."""
    ax = _build_axis(period_x, x_walls, degree, el_x, grade)
    ay = _build_axis(period_y, y_walls, degree, el_y, grade)

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    _require_propagating_incidence("prepare_pmm_2d", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)
    eps_reals = [eps_sup, eps_sub] + [complex(e) for e in
                                      np.asarray(eps_tile).ravel()]

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nf = len(order_x)

    eps0 = eps_tile.flat[0]
    uniform_layer = bool(np.all(np.abs(eps_tile - eps0) < 1e-12))
    Gx0F = Gy0F = EpsF = EinvF = EpnF = IpxF = IpyF = None
    if not uniform_layer:
        # k0-free projected operators (unit derivative ops; the per-wavelength
        # rescale in solve() is GxF = Gx0F/k0 + kx0*IpxF), with exact diagonal
        # handling of wall-less axes (see _scalar_projected_ops).
        lops = _scalar_projected_ops(ax, ay, eps_tile, ox, oy, period_x,
                                     period_y)
        Gx0F, Gy0F = lops["Gx0F"], lops["Gy0F"]
        EpsF, EinvF, EpnF = lops["EpsF"], lops["EinvF"], lops["EpnF"]
        IpxF, IpyF = lops["IpxF"], lops["IpyF"]

    # Incident plane wave (angle + eps_sup, NOT wavelength).
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
            kz_inc0 = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    cinc = np.concatenate([ex0 * delta, ey0 * delta])
    kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))

    return PreparedPMM2D(
        period_x=period_x, period_y=period_y, depth=depth,
        polarization=polarization, formulation=formulation,
        eps_h=eps0, eps_sup=eps_sup, eps_sub=eps_sub,
        uniform_layer=uniform_layer, order_x=order_x, order_y=order_y, Nf=Nf,
        kx0=kx0, ky0=ky0, Gx0F=Gx0F, Gy0F=Gy0F, EpsF=EpsF, EinvF=EinvF,
        EpnF=EpnF, IpxF=IpxF, IpyF=IpyF, cinc=cinc, einc_sq=einc_sq,
        kz_inc=kz_inc, eps_reals=eps_reals)


def prepare_pmm_2d_cell(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
    max_nodal_dof: int = _MAX_NODAL_DOF,
) -> PreparedPMM2D:
    """Assemble the wavelength-INDEPENDENT part of a multi-region cell solve
    once (the :func:`pmm_efficiency_2d_cell` companion of
    :func:`prepare_pmm_2d`).  NON-DISPERSIVE indices and a FIXED
    ``(theta, phi)`` assumed."""
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    x_walls, y_walls, tile = _cell_to_walls_tile(
        eps_cell, period_x, period_y, "prepare_pmm_2d_cell")
    el_x = _axis_elem_counts(period_x, x_walls, degree, elements_per_strip,
                             "prepare_pmm_2d_cell", "x")
    el_y = _axis_elem_counts(period_y, y_walls, degree, elements_per_strip,
                             "prepare_pmm_2d_cell", "y")
    _validate_cell_orders("prepare_pmm_2d_cell", n_orders, degree, el_x, el_y)
    _validate_cell_cost("prepare_pmm_2d_cell", el_x, el_y, degree,
                        max_nodal_dof)
    eps_tile = np.conj(tile)
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    return _prepare_pmm2d_core(
        period_x, period_y, x_walls, y_walls, eps_tile, eps_sup, eps_sub,
        depth, degree, el_x, el_y, grade, polarization, theta, phi, n_orders,
        formulation)


def pmm_efficiency_2d_vs_wavelength(
    period_x: float,
    period_y: float,
    eps_pillar: complex,
    eps_host: complex,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelengths,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
):
    """Hybrid 2-D PMM diffraction efficiencies across a wavelength sweep, reusing
    the geometry-only nodal/pseudo-inverse assembly (the spectral companion to
    :func:`pmm_efficiency_2d`).

    Assembles the wavelength-invariant projected operators ONCE (via
    :func:`prepare_pmm_2d`) and loops only the per-wavelength operator rescale +
    small projected eig + S-matrix.  Largest reuse win when ``degree >>
    n_orders``.  Dispersive media (indices that change with wavelength) are NOT
    reusable -- call :func:`pmm_efficiency_2d` per wavelength instead.

    Returns
    -------
    orders : (Nf, 2) int ndarray
        The (wavelength-independent) retained order pairs ``(m, n)``.
    R, T : (n_wavelengths, Nf) float ndarray
        Reflected / transmitted efficiency per order at each wavelength.
    """
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0:
        raise ValueError("pmm_efficiency_2d_vs_wavelength: wavelengths is "
                         "empty; pass at least one wavelength [m].")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError("pmm_efficiency_2d_vs_wavelength: every wavelength "
                         "must be a finite value > 0 [m].")
    prepared = prepare_pmm_2d(
        period_x, period_y, eps_pillar, eps_host, x_bounds, y_bounds,
        n_substrate, n_superstrate, depth, degree=degree,
        elements_per_strip=elements_per_strip, grade=grade,
        polarization=polarization, theta=theta, phi=phi, n_orders=n_orders,
        formulation=formulation)
    Nf = prepared.Nf
    R = np.empty((wl.size, Nf), dtype=float)
    T = np.empty((wl.size, Nf), dtype=float)
    orders = None
    for i, w in enumerate(wl):
        res = prepared.solve(float(w))
        orders = res[0]
        R[i, :] = np.asarray(res[1])
        T[i, :] = np.asarray(res[2])
    return orders, R, T


def pmm_efficiency_2d_cell_vs_wavelength(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelengths,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
    max_nodal_dof: int = _MAX_NODAL_DOF,
):
    """Multi-region cell efficiencies across a wavelength sweep, reusing the
    geometry-only assembly (the :func:`pmm_efficiency_2d_cell` companion of
    :func:`pmm_efficiency_2d_vs_wavelength`; same returns)."""
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0:
        raise ValueError("pmm_efficiency_2d_cell_vs_wavelength: wavelengths "
                         "is empty; pass at least one wavelength [m].")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError("pmm_efficiency_2d_cell_vs_wavelength: every "
                         "wavelength must be a finite value > 0 [m].")
    prepared = prepare_pmm_2d_cell(
        period_x, period_y, eps_cell, n_substrate, n_superstrate, depth,
        degree=degree, elements_per_strip=elements_per_strip, grade=grade,
        polarization=polarization, theta=theta, phi=phi, n_orders=n_orders,
        formulation=formulation, max_nodal_dof=max_nodal_dof)
    Nf = prepared.Nf
    R = np.empty((wl.size, Nf), dtype=float)
    T = np.empty((wl.size, Nf), dtype=float)
    orders = None
    for i, w in enumerate(wl):
        res = prepared.solve(float(w))
        orders = res[0]
        R[i, :] = np.asarray(res[1])
        T[i, :] = np.asarray(res[2])
    return orders, R, T
