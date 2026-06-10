"""
lumenairy.elements.pmm.stack2d -- multilayer 2-D hybrid PMM (PMM2DStack).
=========================================================================

:class:`PMM2DStack` cascades MULTIPLE doubly-periodic layers -- uniform films,
axis-aligned patterned cells (:func:`pmm_efficiency_2d_cell` geometry), and
in-plane anisotropic tensor cells (:func:`pmm_jones_2d` geometry) -- through
the Redheffer S-matrix in the shared Rayleigh (Fourier) order basis.  It is the
2-D PMM counterpart of :class:`lumenairy.elements.rcwa.RCWAStack` (builder API:
``add_layer`` / ``add_tapered_pillar`` / ``set_source`` / ``solve`` /
``solve_vs_wavelength``).

Because every patterned layer is Fourier-Galerkin PROJECTED into the same
Rayleigh basis, each layer gets its OWN exact spectral-element grid (walls need
not align across layers -- no union-grid constraint, unlike the 1-D
:class:`PMMStack`); the interfaces couple layer modes through the common order
set.  The solve always drives BOTH incident polarizations and returns
``(orders, R(2, N), T(2, N), jones_reflection(2, 2))`` -- the
:func:`rcwa_jones_2d` return shape.

Loss convention: PUBLIC ``Im(eps) > 0`` for loss throughout.
"""
from __future__ import annotations

import numpy as np

from ..rcwa._core import (
    _grazing_safe_wavelength,
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
    _require_propagating_incidence,
)
from ._core import (
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
    _resolve_incidence,
)
from .stack import _warn_stack_energy
from .twod import (
    _C,
    _MAX_NODAL_DOF,
    _axis_elem_counts,
    _build_axis,
    _cell_to_walls_tile,
    _homogeneous_modes,
    _kz_forward2,
    _layer_modes_projected,
    _scalar_projected_ops,
    _validate_cell_cost,
    _validate_cell_orders,
)
from .twod_jones import _require_nonzero_ezz, _tensor_layer_modes

__all__ = ["PMM2DStack"]


class PMM2DStack:
    """Builder for a multilayer doubly-periodic stack solved by the hybrid
    2-D PMM (see the module docstring).

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres); ``period_y`` defaults to ``period_x``.
    n_superstrate, n_substrate : complex, optional
        Half-space refractive indices (isotropic).
    degree, elements_per_strip, grade, n_orders, formulation, max_nodal_dof
        Shared solver knobs, as in :func:`pmm_efficiency_2d_cell` /
        :func:`pmm_jones_2d` (``formulation`` controls the scalar wall-normal
        rule AND the tensor ``E_z`` elimination).
    """

    def __init__(self, period_x, period_y=None, *, n_superstrate=1.0,
                 n_substrate=1.0, degree=11, elements_per_strip=1,
                 grade=False, n_orders=11, formulation="li",
                 max_nodal_dof=_MAX_NODAL_DOF):
        if formulation not in ("li", "laurent"):
            raise ValueError(
                f"PMM2DStack: formulation must be 'li' or 'laurent', got "
                f"{formulation!r}")
        self.period_x = float(period_x)
        self.period_y = float(period_x if period_y is None else period_y)
        self.n_sup = complex(n_superstrate)
        self.n_sub = complex(n_substrate)
        self.degree = int(degree)
        self.n_el = int(elements_per_strip)
        self.grade = bool(grade)
        self.n_orders = int(n_orders)
        self.formulation = formulation
        self.max_nodal_dof = int(max_nodal_dof)
        self._layers = []          # dicts: kind, thickness, payload (PUBLIC eps)
        self._src = None

    # ------------------------------------------------------------------ #
    # builder
    # ------------------------------------------------------------------ #
    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None):
        """Append one layer: a UNIFORM film (``eps``, scalar), a patterned
        scalar cell (``eps_cell``, the :func:`pmm_efficiency_2d_cell` pixel
        grid), or an in-plane anisotropic tensor cell (``eps_tensor_cell``,
        ``(Sx, Sy, 3, 3)``).  Exactly one of the three must be given.

        DISPERSIVE materials (device-geometry roadmap item 5, 2026-06-10):
        any of the three slots may be a ``wl -> value`` callable; such a
        stack is solved with :meth:`solve_vs_wavelength` (a plain
        :meth:`solve` raises)."""
        if float(thickness) <= 0.0 or not np.isfinite(float(thickness)):
            raise ValueError("PMM2DStack.add_layer: thickness must be > 0")
        given = [v is not None for v in (eps, eps_cell, eps_tensor_cell)]
        if sum(given) != 1:
            raise ValueError(
                "PMM2DStack.add_layer: pass exactly ONE of eps (uniform), "
                "eps_cell (scalar pixel grid) or eps_tensor_cell "
                "((Sx, Sy, 3, 3) in-plane tensor grid).")
        for slot, v in (("eps", eps), ("eps_cell", eps_cell),
                        ("eps_tensor_cell", eps_tensor_cell)):
            if callable(v):
                self._layers.append(dict(kind="disp", t=float(thickness),
                                         slot=slot, fn=v))
                return self
        if eps is not None:
            self._layers.append(dict(kind="uniform", t=float(thickness),
                                     eps=complex(eps)))
            return self
        if eps_cell is not None:
            xw, yw, tile = _cell_to_walls_tile(
                eps_cell, self.period_x, self.period_y,
                "PMM2DStack.add_layer")
            if tile.ndim != 2:
                raise ValueError(
                    "PMM2DStack.add_layer: eps_cell must be a scalar (Sx, Sy) "
                    "grid; pass tensor cells via eps_tensor_cell.")
            self._append_patterned("scalar", float(thickness), xw, yw, tile)
            return self
        cell = np.asarray(eps_tensor_cell, dtype=_C)
        if cell.ndim != 4 or cell.shape[2:] != (3, 3):
            raise ValueError(
                f"PMM2DStack.add_layer: eps_tensor_cell must be "
                f"(Sx, Sy, 3, 3), got shape {cell.shape}.")
        xw, yw, tile = _cell_to_walls_tile(
            cell, self.period_x, self.period_y, "PMM2DStack.add_layer")
        _require_nonzero_ezz("PMM2DStack.add_layer", tile)
        # OUT-OF-PLANE tensor layers are allowed (v5.14 roadmap item 3): any
        # such layer promotes the WHOLE cascade to the generalized S-matrix
        # in solve() (the 1-D PMMStack precedent).
        self._append_patterned("tensor", float(thickness), xw, yw, tile)
        return self

    def _append_patterned(self, kind, t, xw, yw, tile):
        el_x = _axis_elem_counts(self.period_x, xw, self.degree, self.n_el,
                                 "PMM2DStack.add_layer", "x")
        el_y = _axis_elem_counts(self.period_y, yw, self.degree, self.n_el,
                                 "PMM2DStack.add_layer", "y")
        _validate_cell_orders("PMM2DStack.add_layer", self.n_orders,
                              self.degree, el_x, el_y)
        _validate_cell_cost("PMM2DStack.add_layer", el_x, el_y, self.degree,
                            self.max_nodal_dof)
        self._layers.append(dict(kind=kind, t=t, xw=list(xw), yw=list(yw),
                                 tile=tile, el_x=el_x, el_y=el_y))

    def add_tapered_pillars(self, thickness, *, pillars, eps_host,
                            n_slices=8):
        """Append MULTI-PILLAR tapered layers as an auto-sliced z-staircase
        (device-geometry roadmap item 1, 2026-06-10) -- the N-feature,
        center-anchored generalization of :meth:`add_tapered_pillar`.

        ``pillars`` is a list of
        ``((cx, cy), (wx_top, wy_top), (wx_bottom, wy_bottom), eps)`` in
        ABSOLUTE metres; each pillar tapers linearly ABOUT ITS OWN FIXED
        CENTER.  ``eps_host`` fills the remainder.  Each slice's walls are
        EXACT spectral-element walls (no pixelation).  Pillars must lie
        strictly inside the cell (no wrap) and may not overlap.
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_pillars: n_slices must be >= 1, got {n_slices}.")
        pil = [((float(c[0]), float(c[1])), (float(wt[0]), float(wt[1])),
                (float(wb[0]), float(wb[1])), complex(e))
               for c, wt, wb, e in pillars]
        eh = complex(eps_host)
        dz = float(thickness) / n
        for k in range(n):
            zeta = (k + 0.5) / n
            rects = []
            for (cx, cy), (wxt, wyt), (wxb, wyb), e in pil:
                wxz = wxt + (wxb - wxt) * zeta
                wyz = wyt + (wyb - wyt) * zeta
                if wxz <= 0.0 or wyz <= 0.0:
                    continue
                x0, x1 = cx - 0.5 * wxz, cx + 0.5 * wxz
                y0, y1 = cy - 0.5 * wyz, cy + 0.5 * wyz
                if not (0.0 < x0 < x1 < self.period_x
                        and 0.0 < y0 < y1 < self.period_y):
                    raise ValueError(
                        "add_tapered_pillars: every pillar must lie strictly "
                        "inside the cell at every slice (no wrap); got "
                        f"x [{x0:.3e}, {x1:.3e}], y [{y0:.3e}, {y1:.3e}].")
                rects.append((x0, x1, y0, y1, e))
            for i, (ax0, ax1, ay0, ay1, _e) in enumerate(rects):
                for bx0, bx1, by0, by1, _e2 in rects[i + 1:]:
                    if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                        raise ValueError(
                            "add_tapered_pillars: pillars overlap; merge or "
                            "separate them explicitly.")
            # INTERIOR wall positions in metres (the _append_patterned
            # convention used by add_tapered_pillar; tile has one strip more
            # than walls per axis)
            xw = sorted({v for r in rects for v in (r[0], r[1])})
            yw = sorted({v for r in rects for v in (r[2], r[3])})
            bx = [0.0] + xw + [self.period_x]
            by = [0.0] + yw + [self.period_y]
            tile = np.full((len(xw) + 1, len(yw) + 1), eh, dtype=complex)
            for ix in range(len(xw) + 1):
                mx = 0.5 * (bx[ix] + bx[ix + 1])
                for iy in range(len(yw) + 1):
                    my = 0.5 * (by[iy] + by[iy + 1])
                    for x0, x1, y0, y1, e in rects:
                        if x0 < mx < x1 and y0 < my < y1:
                            tile[ix, iy] = e
                            break
            self._append_patterned("scalar", dz, xw, yw, tile)
        return self

    def add_tapered_pillar(self, thickness, *, eps_pillar, eps_host,
                           x_bounds_bottom, y_bounds_bottom,
                           x_bounds_top=None, y_bounds_top=None,
                           n_slices=8, rule="midpoint"):
        """Append a TAPERED rectangular pillar (sloped sidewalls) as a
        z-staircase of ``n_slices`` exact-wall scalar layers -- the 2-D
        counterpart of ``RCWAStack.add_tapered_grating``.  The pillar bounds
        interpolate linearly from ``*_bounds_bottom`` to ``*_bounds_top``
        (default: equal -> a straight pillar); each slice's walls are EXACT
        (no pixel snapping).  ``rule='midpoint'`` samples the bounds at slice
        midpoints (O(1/n_slices^2)); ``'bottom'`` at the slice bottoms."""
        if rule not in ("midpoint", "bottom"):
            raise ValueError(
                f"PMM2DStack.add_tapered_pillar: rule must be 'midpoint' or "
                f"'bottom', got {rule!r}")
        n_slices = int(n_slices)
        if n_slices < 1:
            raise ValueError(
                "PMM2DStack.add_tapered_pillar: n_slices must be >= 1")
        xb0 = tuple(map(float, x_bounds_bottom))
        yb0 = tuple(map(float, y_bounds_bottom))
        xb1 = xb0 if x_bounds_top is None else tuple(map(float, x_bounds_top))
        yb1 = yb0 if y_bounds_top is None else tuple(map(float, y_bounds_top))
        dz = float(thickness) / n_slices
        for s in range(n_slices):
            # slice s spans [s*dz, (s+1)*dz] measured from the layer BOTTOM;
            # the stack is built superstrate-first, so slice order is TOP-down
            zfrac = (1.0 - (s + 0.5) / n_slices if rule == "midpoint"
                     else 1.0 - (s + 1.0) / n_slices)
            xw = [xb0[0] + (xb1[0] - xb0[0]) * zfrac,
                  xb0[1] + (xb1[1] - xb0[1]) * zfrac]
            yw = [yb0[0] + (yb1[0] - yb0[0]) * zfrac,
                  yb0[1] + (yb1[1] - yb0[1]) * zfrac]
            if not (0.0 < xw[0] < xw[1] < self.period_x
                    and 0.0 < yw[0] < yw[1] < self.period_y):
                raise ValueError(
                    "PMM2DStack.add_tapered_pillar: interpolated pillar "
                    f"bounds {xw} x {yw} must satisfy 0 < lo < hi < period "
                    "at every slice.")
            tile = np.full((3, 3), complex(eps_host), dtype=_C)
            tile[1, 1] = complex(eps_pillar)
            self._append_patterned("scalar", dz, xw, yw, tile)
        return self

    def plot_geometry(self, axes=None, material_names=None):
        """Draw each layer's exact-wall (x, y) cell map (device-geometry
        roadmap item 7): one panel per layer from the stored analytic walls
        and strip tile -- no pixelation.  Returns the list of axes."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        n = len(self._layers)
        if n == 0:
            raise ValueError("PMM2DStack.plot_geometry: add layers first.")
        if axes is None:
            _fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2),
                                      squeeze=False)
            axes = list(axes[0])
        for ax, L in zip(axes, self._layers):
            kind = L.get("kind")
            if kind == "uniform":
                ax.add_patch(Rectangle((0, 0), self.period_x, self.period_y,
                                       facecolor="0.8"))
                ax.text(0.5 * self.period_x, 0.5 * self.period_y,
                        f"eps={L['eps']:.3g}\nt={L['t']:.3g} m",
                        ha="center", va="center", fontsize=8)
            elif kind == "disp":
                ax.add_patch(Rectangle((0, 0), self.period_x, self.period_y,
                                       facecolor="none", edgecolor="0.4",
                                       hatch="//"))
                ax.text(0.5 * self.period_x, 0.5 * self.period_y,
                        "dispersive", ha="center", va="center", fontsize=8)
            else:
                bx = [0.0] + [float(v) for v in L["xw"]] + [self.period_x]
                by = [0.0] + [float(v) for v in L["yw"]] + [self.period_y]
                tile = np.asarray(L["tile"])
                vals = (np.real(tile[..., 0, 0]) if tile.ndim == 4
                        else np.real(tile))
                vmin, vmax = float(vals.min()), float(vals.max())
                rng = (vmax - vmin) or 1.0
                cmap = plt.get_cmap("viridis")
                for ix in range(len(bx) - 1):
                    for iy in range(len(by) - 1):
                        ax.add_patch(Rectangle(
                            (bx[ix], by[iy]), bx[ix + 1] - bx[ix],
                            by[iy + 1] - by[iy],
                            facecolor=cmap((vals[ix, iy] - vmin) / rng),
                            edgecolor="none"))
                ax.set_title(f"t={L['t']:.3g} m", fontsize=8)
            ax.set_xlim(0, self.period_x)
            ax.set_ylim(0, self.period_y)
            ax.set_aspect("equal")
        return axes

    def set_source(self, wavelength, *, theta=None, phi=0.0, angle=None):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``theta`` / azimuth ``phi`` [rad]).  ``angle`` is the cross-suite alias
        for ``theta`` (theta wins when both are given, as everywhere else)."""
        theta = _resolve_incidence(angle, theta)
        self._src = dict(wavelength=float(wavelength),
                         theta=0.0 if theta is None else float(theta),
                         phi=float(phi))
        return self

    # ------------------------------------------------------------------ #
    # solve
    # ------------------------------------------------------------------ #
    def solve(self):
        """Solve the cascade.  Returns ``(orders, R(2, N), T(2, N),
        jones_reflection(2, 2))`` -- row 0 = incident ``E_x``, row 1 =
        incident ``E_y``; Jones in the PUBLIC convention."""
        if any(L.get("kind") == "disp" for L in self._layers):
            raise ValueError(
                "PMM2DStack.solve: the stack holds DISPERSIVE (wl -> value) "
                "materials; use solve_vs_wavelength(wavelengths), which "
                "materialises every callable per wavelength.")
        if self._src is None:
            raise ValueError("PMM2DStack.solve: call set_source(...) first")
        if not self._layers:
            raise ValueError("PMM2DStack.solve: add at least one layer")
        wavelength = self._src["wavelength"]
        theta, phi = self._src["theta"], self._src["phi"]

        eps_sup = np.conj(_C(self.n_sup) ** 2)
        eps_sub = np.conj(_C(self.n_sub) ** 2)
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("PMM2DStack.solve", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)

        n_orders = self.n_orders
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nf = len(order_x)

        eps_reals = [eps_sup, eps_sub]
        for L in self._layers:
            if L["kind"] == "uniform":
                eps_reals.append(complex(L["eps"]))
            elif L["kind"] == "scalar":
                eps_reals += [complex(e) for e in
                              np.asarray(L["tile"]).ravel()]
            else:
                eps_reals += [complex(e) for e in
                              np.asarray(L["tile"][..., [0, 1, 2],
                                                   [0, 1, 2]]).ravel()]
        wl = _grazing_safe_wavelength(wavelength, kx0, ky0, order_x, order_y,
                                      self.period_x, self.period_y, eps_reals)
        k0 = 2.0 * np.pi / wl
        kxv = kx0 + order_x * (wl / self.period_x)
        kyv = ky0 + order_y * (wl / self.period_y)

        Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
        Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

        # per-layer modes in the shared Rayleigh basis
        modes = []
        for L in self._layers:
            if L["kind"] == "uniform":
                Wl, Vl, lam, _ = _homogeneous_modes(kxv, kyv,
                                                    np.conj(_C(L["eps"])))
            elif L["kind"] == "scalar":
                tile_i = np.conj(L["tile"])
                eps0 = tile_i.flat[0]
                if bool(np.all(np.abs(tile_i - eps0) < 1e-12)):
                    Wl, Vl, lam, _ = _homogeneous_modes(kxv, kyv, eps0)
                else:
                    ax = _build_axis(self.period_x, L["xw"], self.degree,
                                     L["el_x"], self.grade)
                    ay = _build_axis(self.period_y, L["yw"], self.degree,
                                     L["el_y"], self.grade)
                    lops = _scalar_projected_ops(ax, ay, tile_i, ox, oy,
                                                 self.period_x, self.period_y)
                    GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
                    GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
                    Wl, Vl, lam = _layer_modes_projected(
                        GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
                        formulation=self.formulation)
            else:                                   # tensor (full 3x3)
                tile_i = np.conj(L["tile"])
                ax = _build_axis(self.period_x, L["xw"], self.degree,
                                 L["el_x"], self.grade)
                ay = _build_axis(self.period_y, L["yw"], self.degree,
                                 L["el_y"], self.grade)
                ez_rule = ("li" if self.formulation == "li" else "laurent")
                out = _tensor_layer_modes(
                    ax, ay, L["xw"], L["yw"], tile_i, k0, kx0, ky0, ox, oy,
                    kxv, kyv, ez_rule)
                if len(out) == 6:                   # out-of-plane generator
                    modes.append(("gen",) + out + (L["t"],))
                    continue
                Wl, Vl, lam = out
            modes.append(("sym", Wl, Vl, lam, L["t"]))

        any_oop = any(m[0] == "gen" for m in modes)
        if not any_oop:
            # Redheffer cascade: sup | L1 | L2 | ... | Ln | sub (symmetric)
            W_prev, V_prev = Wsup, Vsup
            S = None
            for (_k, Wl, Vl, lam, t) in modes:
                Si = _interface_smatrix(W_prev, V_prev, Wl, Vl)
                S = Si if S is None else _redheffer_star(S, Si)
                S = _redheffer_star(S, _propagation_smatrix(lam, k0 * t))
                W_prev, V_prev = Wl, Vl
            S = _redheffer_star(S, _interface_smatrix(W_prev, V_prev, Wsub,
                                                      Vsub))
        else:
            # GENERALIZED cascade (v5.14): any out-of-plane layer breaks the
            # [W; -V] <-> -lam symmetry, so the whole stack is promoted --
            # symmetric layers/half-spaces enter as [W, W; V, -V] blocks with
            # (lam, -lam), the generator layers with their explicit
            # forward/backward sets (the 1-D PMMStack precedent).
            def _blocks(m):
                if m[0] == "sym":
                    _k, W, V, lam, t = m
                    return _modes_to_M(W, V, W, -V), lam, -lam, t
                _k, Wf, Vf, lf, Wb, Vb, lb, t = m
                return _modes_to_M(Wf, Vf, Wb, Vb), lf, lb, t

            M_prev = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
            S = None
            for m in modes:
                Ml, lf, lb, t = _blocks(m)
                Si = _interface_smatrix_general(M_prev, Ml)
                S = Si if S is None else _redheffer_star(S, Si)
                S = _redheffer_star(
                    S, _propagation_smatrix_general(lf, lb, k0 * t))
                M_prev = Ml
            Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
            S = _redheffer_star(S, _interface_smatrix_general(M_prev, Msub))
        S11, _S12, S21, _S22 = S

        # Jones far field (both incident polarizations)
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
        kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
        kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
        safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
        safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
        R_rows, T_rows, j_cols = [], [], []
        for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
            long_inc = (kx0 * ex0 + ky0 * ey0)
            einc_sq = (1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0
                       else 1.0)
            cinc = np.concatenate([ex0 * delta, ey0 * delta])
            r = S11 @ cinc
            t_ = S21 @ cinc
            rx, ry = r[:Nf], r[Nf:]
            tx, ty = t_[:Nf], t_[Nf:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re = np.real(kz_ref_f / kz_inc) * (
                np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) / einc_sq
            Te = np.real(kz_trn_f / kz_inc) * (
                np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) / einc_sq
            R_rows.append(np.where(np.real(kz_ref_f) > 0, np.real(Re), 0.0))
            T_rows.append(np.where(np.real(kz_trn_f) > 0, np.real(Te), 0.0))
            j_cols.append(np.stack([np.conj(rx[p0]), np.conj(ry[p0])]))
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jones = np.stack(j_cols, axis=1)
        orders2d = np.stack([order_x, order_y], axis=1)
        _warn_stack_energy(R_eff, T_eff)
        return orders2d, R_eff, T_eff, jones

    def _materialized_layers(self, w, layers):
        """Concrete layer list at one wavelength: each DISPERSIVE layer's
        callable is resolved and run through the normal add-time processing
        (walls/tile extraction + validation); concrete layers pass through."""
        out = []
        for L in layers:
            if L.get("kind") != "disp":
                out.append(L)
                continue
            probe = PMM2DStack.__new__(PMM2DStack)
            probe.__dict__.update(self.__dict__)
            probe._layers = []
            probe.add_layer(L["t"], **{L["slot"]: L["fn"](float(w))})
            out.extend(probe._layers)
        return out

    def solve_vs_wavelength(self, wavelengths, *, theta=None, phi=0.0,
                            angle=None, jones=False):
        """Solve the stack across a wavelength sweep, materialising any
        DISPERSIVE (``wl -> value``) layer callables per wavelength (item 5,
        device-geometry roadmap 2026-06-10).
        Returns ``(orders, R(n_wl, 2, N), T(n_wl, 2, N))``, plus a FOURTH
        ``jones (n_wl, 2, 2)`` element when ``jones=True`` (default ``False``
        keeps the released 3-tuple)."""
        wlv = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        if wlv.size == 0:
            raise ValueError("PMM2DStack.solve_vs_wavelength: wavelengths is "
                             "empty; pass at least one wavelength [m].")
        if not np.all(np.isfinite(wlv)) or np.any(wlv <= 0.0):
            raise ValueError("PMM2DStack.solve_vs_wavelength: every "
                             "wavelength must be a finite value > 0 [m].")
        base = self._layers
        orders = R = T = J = None
        try:
            for i, w in enumerate(wlv):
                self._layers = self._materialized_layers(float(w), base)
                self.set_source(float(w), theta=theta, phi=phi, angle=angle)
                o, R1, T1, j1 = self.solve()
                if orders is None:
                    orders = o
                    R = np.empty((wlv.size,) + R1.shape, dtype=float)
                    T = np.empty((wlv.size,) + T1.shape, dtype=float)
                    J = np.empty((wlv.size, 2, 2), dtype=complex)
                R[i] = R1
                T[i] = T1
                J[i] = np.asarray(j1)
        finally:
            self._layers = base
        if jones:
            return orders, R, T, J
        return orders, R, T
