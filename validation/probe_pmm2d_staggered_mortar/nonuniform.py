"""F5 (roadmap item N-1) -- NON-UNIFORM SEGMENT BOUNDARIES for the staggered
modified-Legendre basis.  PROTOTYPE, not library code.

The shipped :class:`lumenairy.elements.pmm.twod_staggered.Basis1D` hard-codes a
UNIFORM lattice::

    self.h  = self.d / self.N
    self.J  = 0.5 * self.h                    # ONE scalar jacobian
    self.xb = np.linspace(0.0, self.d, self.N + 1)

but Granet 2023 Eq. 31 does NOT::

    "consider a line I of length d divided by N adjacent segments
     I_n = [x_n, x_{n+1}], n = 1, 2, ... N, and I = U I_n.  Each segment is
     mapped to the reference interval [-1, 1] by the change of variable
         x = 0.5 (x_{n+1} - x_n) u + 0.5 (x_{n+1} + x_n)."          (Eq. 31)

Non-uniform segments are already in the formulation; the uniform lattice is an
implementation choice.  Generalising it needs exactly four edits, all of them
the SAME edit -- replace the scalar ``J`` by a per-segment ``J_n``:

1. ``Basis1D._global_matrix``    mass ``* J`` -> ``* J_n``;  stiffness
                                 ``/ J`` -> ``/ J_n``;  mixed ``* 1`` unchanged
                                 (the ``du`` of the integral cancels the
                                 ``1/J_n`` of ``d/dx`` on EVERY segment, so the
                                 mixed matrix was already non-uniform-correct).
2. ``_global_pair_segmat``       the same, per segment.
3. ``Granet2DTransverseE._eps_dir``'s inline ``segmat`` -- the same.
4. ``_stag_fourier_projection``  ``xphys = mid_n + J_n u`` and the ``J_n / d``
                                 weight.

Everything else is INVARIANT and that is the reason this is cheap:

* the elementary matrices ``m_ref`` / ``s_ref`` / ``c_ref`` live on the
  REFERENCE interval and never saw ``J``;
* the hat functions (Eq. 32) glue ``Ltilde_2`` of one segment to ``Ltilde_1``
  of the next by VALUE (``Ltilde_1(-1) = 1``, ``Ltilde_2(+1) = 1``,
  ``Ltilde_1(+1) = Ltilde_2(-1) = 0``), which is a statement about the
  reference interval alone -- so the Bloch periodic hat (Eq. 33, the ``tau``
  glue) is unchanged;
* the de Rham property ``d(Btilde) subset span(B)`` -- the thing that makes the
  staggered basis spurious-free -- is per-segment and scale-free: on segment
  ``n`` a ``Btilde`` member is a polynomial of degree ``<= M-1``, its
  derivative has degree ``<= M-2``, and ``B``'s local span IS every polynomial
  of degree ``<= M-2``.  Multiplying by ``1/J_n`` does not leave that span.
* the mortar cross-mass :func:`mortar2d.cross_mass_1d` is ALREADY general: it
  integrates on the union of the two partitions and maps each union
  sub-interval into each side's own segment with that segment's own affine map.

Public entry points
-------------------
``Basis1DNU(d, walls, M, tau)``   ``walls`` = an int ``N`` (uniform, and then
                                  BIT-IDENTICAL to the shipped basis) or an
                                  array of ``N+1`` boundaries.
``Granet2DTransverseE_NU``        the shipped eigensolver on two ``Basis1DNU``.
``MortarStackNU``                 the per-layer-grid cascade with per-layer
                                  WALL SETS -- the taper path.
"""
from __future__ import annotations

import time

import numpy as np
from numpy.polynomial.legendre import leggauss

from mortar2d import (CrossOps, guard, interface_general_mortar_2d,
                      interface_mortar_2d, kron_apply)

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
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Basis1D,
    Granet2DTransverseE,
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

__all__ = ["Basis1DNU", "Granet2DTransverseE_NU", "MortarStackNU",
           "far_projector_2d_nu", "stag_fourier_projection_nu",
           "global_pair_segmat_nu"]


# =========================================================================== #
class Basis1DNU(Basis1D):
    """:class:`Basis1D` on ARBITRARY segment boundaries (Granet Eq. 31).

    ``walls`` is either an ``int`` ``N`` -- the uniform lattice, reproducing
    the shipped basis BIT-FOR-BIT -- or a length ``N+1`` increasing array of
    boundaries with ``walls[0] == 0`` and ``walls[-1] == d``.

    ``self.J`` (the scalar jacobian) is ``None`` on a non-uniform basis ON
    PURPOSE: any shipped code path that still reads it fails loudly with a
    ``TypeError`` instead of silently applying one segment's scaling to all of
    them.  The per-segment jacobians are ``self.Jn``.
    """

    def __init__(self, d, walls, M, tau=1.0 + 0.0j):
        assert M >= 3, "Basis1D needs M>=3 (M=2 gives a degenerate cardinality)"
        self.d = float(d)
        self.M = int(M)
        self.tau = _C(tau)
        if np.ndim(walls) == 0:
            self.N = int(walls)
            self.h = self.d / self.N
            self.J = 0.5 * self.h                    # shipped scalar
            self.xb = np.linspace(0.0, self.d, self.N + 1)
            self.Jn = np.full(self.N, self.J, dtype=float)
            self.uniform = True
        else:
            xb = np.asarray(walls, dtype=float)
            if xb.ndim != 1 or xb.size < 2:
                raise ValueError("walls must be an int N or an (N+1,) array")
            if abs(xb[0]) > 1e-15 * self.d or abs(xb[-1] - self.d) > 1e-13 * self.d:
                raise ValueError(f"walls must run 0 .. d, got {xb[0]}..{xb[-1]}")
            if np.any(np.diff(xb) <= 0):
                raise ValueError("walls must be strictly increasing")
            self.N = xb.size - 1
            self.xb = xb
            self.Jn = 0.5 * np.diff(xb)
            self.h = None
            self.J = None                            # loud, not silent
            self.uniform = bool(np.allclose(np.diff(xb), self.d / self.N,
                                            rtol=0, atol=1e-15 * self.d))
        self._build_elementary()
        self._build_sets()

    # ---- the ONE generalization: per-segment physical scaling --------------
    def _seg_scale(self, ref):
        """Physical scaling per segment for an elementary reference matrix."""
        if ref is self.m_ref:
            return self.Jn                            # mass:      * J_n
        if ref is self.s_ref:
            return 1.0 / self.Jn                      # stiffness: / J_n
        return np.ones(self.N)                        # one derivative: * 1

    def _global_matrix(self, ref, setL, setR, eps_seg=None):
        w_seg = np.asarray(self._seg_scale(ref), dtype=_C)
        if eps_seg is not None:
            w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
        L_ten = np.array(setL)
        R_ten = np.array(setR)
        RR = np.einsum("ab,jsb->jsa", ref, R_ten)
        return np.einsum("isa,s,jsa->ij", np.conj(L_ten), w_seg, RR)


def global_pair_segmat_nu(basis: Basis1DNU, ref, setL, setR):
    """Per-segment contributions, non-uniform twin of ``_global_pair_segmat``."""
    L_ten = np.array(setL)
    R_ten = np.array(setR)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    G0 = np.einsum("isa,jsa->sij", np.conj(L_ten), RR)
    return np.asarray(basis._seg_scale(ref))[:, None, None] * G0


def stag_fourier_projection_nu(basis: Basis1DNU, orders, alpha0=0.0):
    """Non-uniform twin of ``_stag_fourier_projection``: the ONLY change is
    that the quadrature points and the ``J/d`` weight are built from segment
    ``n``'s own affine map (Eq. 31) instead of one shared ``J``."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        J = basis.Jn[seg]
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

    def _assemble(global_set):
        S = np.array(global_set)
        return np.einsum("msa,jsa->mj", T_local, S)
    return _assemble


def far_projector_2d_nu(bx, by, ox, oy, alpha0x=0.0, alpha0y=0.0):
    asmx = stag_fourier_projection_nu(bx, ox, alpha0x)
    asmy = stag_fourier_projection_nu(by, oy, alpha0y)
    P1 = np.kron(asmy(by.Btilde), asmx(bx.B))
    P2 = np.kron(asmy(by.B), asmx(bx.Btilde))
    return P1, P2


# =========================================================================== #
class Granet2DTransverseE_NU(Granet2DTransverseE):
    """The shipped staggered eigensolver on two :class:`Basis1DNU` axes.

    ``wx`` / ``wy`` are per-axis segment boundaries (or ints).  ``eps_cell`` is
    the per-SEGMENT-CELL map, shape ``(Nx, Ny)`` or ``(Nx, Ny, 3, 3)`` -- i.e.
    exactly today's cell, except that the cell's grid is no longer forced to be
    the uniform lattice: the WALLS are now an independent input.
    """

    def __init__(self, px, py, wx, wy, M, eps_cell,
                 alpha0x=0.0, alpha0y=0.0, k0=2.0 * np.pi):
        self.k0 = float(k0)
        self.alpha0x = float(alpha0x)
        self.alpha0y = float(alpha0y)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1DNU(px, wx, M, taux)
        self.by = Basis1DNU(py, wy, M, tauy)
        self.eps_cell = np.asarray(eps_cell, dtype=_C)
        if self.eps_cell.ndim not in (2, 4) or (
                self.eps_cell.ndim == 4 and self.eps_cell.shape[2:] != (3, 3)):
            raise ValueError("eps_cell must be (Nx, Ny) or (Nx, Ny, 3, 3)")
        if self.eps_cell.shape[:2] != (self.bx.N, self.by.N):
            raise ValueError(
                f"eps_cell {self.eps_cell.shape[:2]} does not match the wall "
                f"sets ({self.bx.N}, {self.by.N})")
        self.q = self.bx.dim
        assert self.bx.dim == self.by.dim, "use square (Nx == Ny)"
        from lumenairy.elements.pmm.twod_staggered import _tile_is_offplane
        self.offplane = (self.eps_cell.ndim == 4
                         and _tile_is_offplane(self.eps_cell))
        if self.offplane:
            self._assemble_oop()
        else:
            self._assemble()

    # the two assembly helpers that carry the scalar J -- per-segment now
    def _eps_weighted(self, refx_pair, refy_pair, wmap=None):
        bx, refx, sLx, sRx = refx_pair
        by, refy, sLy, sRy = refy_pair
        Gx = global_pair_segmat_nu(bx, refx, sLx, sRx)
        Gy = global_pair_segmat_nu(by, refy, sLy, sRy)
        eps = self.eps_cell if wmap is None else wmap
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out

    def _eps_dir(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
        def segmat(basis, lset, op, rset):
            Lt = np.array(getattr(basis, lset))
            Rt = np.array(getattr(basis, rset))
            if op == "m":
                RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
                scale = np.asarray(basis.Jn)
            elif op == "dL":
                RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
                scale = np.ones(basis.N)
            else:
                RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
                scale = np.ones(basis.N)
            return scale[:, None, None] * np.einsum("isa,jsa->sij",
                                                    np.conj(Lt), RR)
        Gx = segmat(bx, lx, opx, rx)
        Gy = segmat(by, ly, opy, ry)
        eps = self.eps_cell if wmap is None else wmap
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out


# =========================================================================== #
class GridOpsNU:
    """Per-grid geometry keyed on the WALL SET (not on an integer ``N``)."""

    def __init__(self, px, py, wx, wy, M, taux, tauy):
        self.M = int(M)
        self.bx = Basis1DNU(px, wx, M, taux)
        self.by = Basis1DNU(py, wy, M, tauy)
        self.N = self.bx.N
        self.q = self.bx.dim
        self.qq = self.q * self.q
        self.Mtt_x = self.bx.mass(self.bx.Btilde, self.bx.Btilde)
        self.Mbb_x = self.bx.mass(self.bx.B, self.bx.B)
        self.Mtt_y = self.by.mass(self.by.Btilde, self.by.Btilde)
        self.Mbb_y = self.by.mass(self.by.B, self.by.B)
        self.V1 = (self.Mtt_y, self.Mbb_x)
        self.V2 = (self.Mbb_y, self.Mtt_x)
        self._key = (self.bx.xb.tobytes(), self.by.xb.tobytes(), self.M)

    def key(self):
        return self._key


def _wall_array(period, walls, n_default):
    """``walls`` -> a full boundary array on [0, period]."""
    if walls is None:
        return np.linspace(0.0, period, n_default + 1)
    w = np.asarray(walls, dtype=float).ravel()
    if w.size and (abs(w[0]) < 1e-15 * period) and \
            abs(w[-1] - period) < 1e-13 * period:
        return w
    return np.concatenate([[0.0], w, [period]])       # INTERIOR walls given


class MortarStackNU:
    """Per-layer NON-UNIFORM-grid twin of ``MortarStack2D``.

    ``add_layer(thickness, eps_cell=..., x_walls=..., y_walls=..., n_modes=)``
    -- ``x_walls`` / ``y_walls`` are the layer's own INTERIOR wall positions
    (or the full ``0..period`` boundary array).  Omitted, they default to the
    uniform lattice implied by ``eps_cell.shape``, i.e. today's behaviour.
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

    def add_layer(self, thickness, *, eps=None, eps_cell=None, x_walls=None,
                  y_walls=None, grid=None, n_modes=None):
        if (eps is None) == (eps_cell is None):
            raise ValueError("pass exactly one of eps / eps_cell")
        M = int(self.M if n_modes is None else n_modes)
        if eps_cell is not None:
            cell = np.asarray(eps_cell, dtype=_C)
            wx = _wall_array(self.period_x, x_walls, cell.shape[0])
            wy = _wall_array(self.period_y, y_walls, cell.shape[1])
            self._layers.append(dict(kind="patterned", thickness=float(thickness),
                                     eps_cell=cell, wx=wx, wy=wy, M=M))
        else:
            e = np.asarray(eps, dtype=_C)
            if x_walls is None and y_walls is None and grid is None:
                grid = self._layers[-1]["wx"].size - 1 if self._layers else 1
            wx = _wall_array(self.period_x, x_walls, grid or 1)
            wy = _wall_array(self.period_y, y_walls, grid or 1)
            kind = "uniform" if e.ndim == 0 else "uniform_tensor"
            self._layers.append(dict(kind=kind, thickness=float(thickness),
                                     eps=_C(eps) if e.ndim == 0 else None,
                                     eps33=None if e.ndim == 0 else e,
                                     wx=wx, wy=wy, M=M))
        return self

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        self._src = dict(wl=float(wavelength), theta=float(theta),
                         phi=float(phi))
        return self

    def solve(self, *, jones=True, force_general=False, force_mortar=False):
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

        grids = {}

        def _grid(wx, wy, M):
            key = (wx.tobytes(), wy.tobytes(), M)
            g = grids.get(key)
            if g is None:
                g = GridOpsNU(px, py, wx, wy, M, taux, tauy)
                grids[key] = g
            return g

        gof = [_grid(L["wx"], L["wy"], L["M"]) for L in self._layers]
        g_sup, g_sub = gof[0], gof[-1]

        geo_cache = {}

        def _geo(g):
            hit = geo_cache.get(g.key())
            if hit is None:
                sol = Granet2DTransverseE_NU(
                    px, py, g.bx.xb, g.by.xb, g.M,
                    np.full((g.bx.N, g.by.N), eps_sup),
                    alpha0x=a0x, alpha0y=a0y, k0=k0)
                hit = _homog_geom_cache(sol)
                geo_cache[g.key()] = hit
            return hit

        t0 = time.perf_counter()
        Wsup, Vsup, lam_sup = _homog_region_modes(_geo(g_sup), eps_sup)
        Wsub, Vsub, lam_sub = _homog_region_modes(_geo(g_sub), eps_sub)
        self.timing["halfspace_eig"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        modes, any_oop, eig_cache = [], False, {}
        for L, g in zip(self._layers, gof):
            if L["kind"] == "uniform":
                W, V, lam = _homog_region_modes(_geo(g), L["eps"])
                six = _modes_as_general(W, V, lam)
            else:
                if L["kind"] == "uniform_tensor":
                    cell = np.ascontiguousarray(
                        np.broadcast_to(L["eps33"], (g.bx.N, g.by.N, 3, 3)))
                else:
                    cell = L["eps_cell"]
                key = (g.key(), cell.shape, cell.tobytes())
                cached = eig_cache.get(key)
                if cached is None:
                    sol = Granet2DTransverseE_NU(px, py, g.bx.xb, g.by.xb,
                                                 g.M, cell, alpha0x=a0x,
                                                 alpha0y=a0y, k0=k0)
                    if sol.offplane:
                        cached = _region_modes_oop(sol)
                    else:
                        Wl, Vl, lam_l, _g2 = _region_modes(sol)
                        cached = _modes_as_general(Wl, Vl, lam_l)
                    eig_cache[key] = cached
                six = cached
                if cell.ndim == 4 and _tile_needs_oop("MortarStackNU", cell):
                    any_oop = True
            modes.append(six + (L["thickness"],))
        self.timing["layer_eig"] = time.perf_counter() - t0
        general = any_oop or force_general

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

        cap = min((g_sup.q - 1) // 2, (g_sub.q - 1) // 2)
        n_orders = min(self.n_orders, cap)
        self.n_orders_used = n_orders
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nfo = len(order_x)
        P1s, P2s = far_projector_2d_nu(g_sup.bx, g_sup.by, ox, oy, a0x, a0y)
        P1t, P2t = far_projector_2d_nu(g_sub.bx, g_sub.by, ox, oy, a0x, a0y)
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
            cinc = _guarded_lstsq(Hsup, rhs, "MortarStackNU far-field")
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


# keep the linter honest about the re-exported helper
_ = kron_apply
