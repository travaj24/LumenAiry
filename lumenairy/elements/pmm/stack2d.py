"""
lumenairy.elements.pmm.stack2d -- multilayer 2-D hybrid PMM (PMM2DStackHybrid).
=========================================================================

:class:`PMM2DStackHybrid` cascades MULTIPLE doubly-periodic layers -- uniform films,
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

import copy as _copy
import os
import warnings

import numpy as np

from ..rcwa._core import (
    _blas_threads_quiet,
    _grazing_safe_wavelength,
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
    _require_propagating_incidence,
    _symmetry_on,
)
from ._core import (
    PerOrderAmplitudesMixin,
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

__all__ = ["PMM2DStackHybrid", "PMM2DStack_hybrid", "PMM2DStack"]


from ...backend import is_jax_array


class PMM2DStackHybrid(PerOrderAmplitudesMixin):
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
                 symmetry="auto", max_nodal_dof=_MAX_NODAL_DOF):
        if formulation not in ("li", "laurent"):
            raise ValueError(
                f"PMM2DStackHybrid: formulation must be 'li' or 'laurent', got "
                f"{formulation!r}")
        self.period_x = float(period_x)
        self.period_y = float(period_x if period_y is None else period_y)
        # A JAX half-space index stays RAW so its gradient flows (the
        # differentiable dispatch in solve()); complex() would sever it.
        self.n_sup = (n_superstrate if is_jax_array(n_superstrate)
                      else complex(n_superstrate))
        self.n_sub = (n_substrate if is_jax_array(n_substrate)
                      else complex(n_substrate))
        self.degree = int(degree)
        self.n_el = int(elements_per_strip)
        self.grade = bool(grade)
        self.n_orders = int(n_orders)
        self.formulation = formulation
        # F2 (audit): even-parity fold for the cascade.  'auto' (the default;
        # True is equivalent) folds when the precondition holds -- normal
        # incidence + a centro-symmetric scalar stack -> the whole Redheffer
        # recursion runs in the (Nf+1)-d EVEN sector (~2-8x on the O(Nf^3)
        # eig/S-matrix steps), with a per-layer flip-invariance guard falling
        # back to the full solve.  symmetry=False forces the full solve (the
        # even basis matches it to ~1e-12, not bit-for-bit).
        self.symmetry = _symmetry_on(symmetry)
        self.max_nodal_dof = int(max_nodal_dof)
        self._layers = []          # dicts: kind, thickness, payload (PUBLIC eps)
        self._src = None
        # F4 part 2 (audit): content-keyed cache of the WAVELENGTH-INDEPENDENT
        # per-layer build (ax/ay nodal assembly + the k0-free projected
        # operators) so a wavelength sweep reuses it instead of rebuilding it
        # per layer per wl.  Persists across solve() calls; cleared on add_layer
        # (geometry change).  A dispersive layer's tile changes per wl -> a new
        # content key, so the cache never serves a stale build.
        self._geom_cache = {}

    # ------------------------------------------------------------------ #
    # builder
    # ------------------------------------------------------------------ #
    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None, region_layout=None):
        """Append one layer: a UNIFORM film (``eps``, scalar), a patterned
        scalar cell (``eps_cell``, the :func:`pmm_efficiency_2d_cell` pixel
        grid), or an in-plane anisotropic tensor cell (``eps_tensor_cell``,
        ``(Sx, Sy, 3, 3)``).  Exactly one of the three must be given.

        DISPERSIVE materials (device-geometry roadmap item 5, 2026-06-10):
        any of the three slots may be a ``wl -> value`` callable; such a
        stack is solved with :meth:`solve_vs_wavelength` (a plain
        :meth:`solve` raises).

        DIFFERENTIABLE (JAX): ``eps`` and ``thickness`` may be traced jnp
        scalars.  A traced ``eps_cell`` additionally needs
        ``region_layout=`` -- a CONCRETE int grid (same shape) naming which
        pixels share a region: the walls come from the layout (wall
        extraction from pixel values is data-dependent control flow a
        tracer cannot drive), the traced cell provides each region's value,
        read at the region's first pixel.  Traced ``eps_tensor_cell``
        raises (the 2-D JAX surface is scalar)."""
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        self._geom_cache = {}   # F4 part 2: geometry changed -> drop the cache
        if is_jax_array(thickness):
            t_store = thickness            # traced: validated only if concrete
            try:
                t_c = float(np.asarray(thickness))
            except Exception:
                t_c = None
            if t_c is not None and (t_c <= 0.0 or not np.isfinite(t_c)):
                raise ValueError("PMM2DStackHybrid.add_layer: thickness must be > 0")
        else:
            if float(thickness) <= 0.0 or not np.isfinite(float(thickness)):
                raise ValueError("PMM2DStackHybrid.add_layer: thickness must be > 0")
            t_store = float(thickness)
        given = [v is not None for v in (eps, eps_cell, eps_tensor_cell)]
        if sum(given) != 1:
            raise ValueError(
                "PMM2DStackHybrid.add_layer: pass exactly ONE of eps (uniform), "
                "eps_cell (scalar pixel grid) or eps_tensor_cell "
                "((Sx, Sy, 3, 3) in-plane tensor grid).")
        for slot, v in (("eps", eps), ("eps_cell", eps_cell),
                        ("eps_tensor_cell", eps_tensor_cell)):
            if callable(v):
                self._layers.append(dict(kind="disp", t=float(thickness),
                                         slot=slot, fn=v))
                return self
        if eps is not None:
            self._layers.append(dict(
                kind="uniform", t=t_store,
                eps=eps if is_jax_array(eps) else complex(eps)))
            return self
        if eps_cell is not None:
            if is_jax_array(eps_cell):
                # TRACED cell values: a CONCRETE region_layout (int grid,
                # same shape) must define the walls -- wall extraction from
                # pixel values is data-dependent control flow a tracer
                # cannot drive (the _pmm_efficiency_2d_cell_jax contract).
                if region_layout is None:
                    raise ValueError(
                        "PMM2DStackHybrid.add_layer: a JAX (traced) eps_cell needs "
                        "region_layout= (a CONCRETE int grid, same shape, "
                        "naming which pixels share a region); the walls come "
                        "from the layout, the traced cell provides each "
                        "region's value.")
                lay = np.ascontiguousarray(np.asarray(region_layout,
                                                      dtype=np.int64))
                if lay.shape != tuple(eps_cell.shape):
                    raise ValueError(
                        "PMM2DStackHybrid.add_layer: region_layout shape "
                        f"{lay.shape} != eps_cell shape "
                        f"{tuple(eps_cell.shape)}.")
                xw, yw, ltile = _cell_to_walls_tile(
                    lay.astype(complex), self.period_x, self.period_y,
                    "PMM2DStackHybrid.add_layer")
                first_idx = [tuple(int(v) for v in np.argwhere(lay == u)[0])
                             for u in np.unique(lay)]
                self._append_patterned("scalar", t_store, xw, yw,
                                       np.real(ltile).astype(np.int64))
                self._layers[-1].update(traced=True, cell=eps_cell,
                                        layout_tile=self._layers[-1].pop(
                                            "tile"),
                                        first_idx=first_idx)
                return self
            if region_layout is not None:
                raise ValueError(
                    "PMM2DStackHybrid.add_layer: region_layout is only meaningful "
                    "for a JAX (traced) eps_cell.")
            xw, yw, tile = _cell_to_walls_tile(
                eps_cell, self.period_x, self.period_y,
                "PMM2DStackHybrid.add_layer")
            if tile.ndim != 2:
                raise ValueError(
                    "PMM2DStackHybrid.add_layer: eps_cell must be a scalar (Sx, Sy) "
                    "grid; pass tensor cells via eps_tensor_cell.")
            self._append_patterned("scalar", t_store, xw, yw, tile)
            return self
        if is_jax_array(eps_tensor_cell):
            raise NotImplementedError(
                "PMM2DStackHybrid.add_layer: traced eps_tensor_cell is not "
                "differentiable (the 2-D JAX surface is scalar; tensor "
                "cells are NumPy-only).")
        cell = np.asarray(eps_tensor_cell, dtype=_C)
        if cell.ndim != 4 or cell.shape[2:] != (3, 3):
            raise ValueError(
                f"PMM2DStackHybrid.add_layer: eps_tensor_cell must be "
                f"(Sx, Sy, 3, 3), got shape {cell.shape}.")
        xw, yw, tile = _cell_to_walls_tile(
            cell, self.period_x, self.period_y, "PMM2DStackHybrid.add_layer")
        _require_nonzero_ezz("PMM2DStackHybrid.add_layer", tile)
        # OUT-OF-PLANE tensor layers are allowed (v5.14 roadmap item 3): any
        # such layer promotes the WHOLE cascade to the generalized S-matrix
        # in solve() (the 1-D PMMStack precedent).
        self._append_patterned("tensor", t_store, xw, yw, tile)
        return self

    def _append_patterned(self, kind, t, xw, yw, tile):
        el_x = _axis_elem_counts(self.period_x, xw, self.degree, self.n_el,
                                 "PMM2DStackHybrid.add_layer", "x")
        el_y = _axis_elem_counts(self.period_y, yw, self.degree, self.n_el,
                                 "PMM2DStackHybrid.add_layer", "y")
        _validate_cell_orders("PMM2DStackHybrid.add_layer", self.n_orders,
                              self.degree, el_x, el_y)
        _validate_cell_cost("PMM2DStackHybrid.add_layer", el_x, el_y, self.degree,
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
                f"PMM2DStackHybrid.add_tapered_pillar: rule must be 'midpoint' or "
                f"'bottom', got {rule!r}")
        n_slices = int(n_slices)
        if n_slices < 1:
            raise ValueError(
                "PMM2DStackHybrid.add_tapered_pillar: n_slices must be >= 1")
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
                    "PMM2DStackHybrid.add_tapered_pillar: interpolated pillar "
                    f"bounds {xw} x {yw} must satisfy 0 < lo < hi < period "
                    "at every slice.")
            tile = np.full((3, 3), complex(eps_host), dtype=_C)
            tile[1, 1] = complex(eps_pillar)
            self._append_patterned("scalar", dz, xw, yw, tile)
        return self

    def plot_geometry(self, axes=None, material_names=None):
        """Draw each layer's exact-wall (x, y) cell map (device-geometry
        roadmap item 7): one panel per layer from the stored analytic walls
        and strip tile -- no pixelation.

        Returns
        -------
        list of matplotlib Axes -- use ``axes[0].figure.savefig(...)`` to
        save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        n = len(self._layers)
        if n == 0:
            raise ValueError("PMM2DStackHybrid.plot_geometry: add layers first.")
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
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        theta = _resolve_incidence(angle, theta)
        self._src = dict(
            wavelength=(wavelength if is_jax_array(wavelength)
                        else float(wavelength)),
            theta=(0.0 if theta is None
                   else theta if is_jax_array(theta) else float(theta)),
            phi=phi if is_jax_array(phi) else float(phi))
        return self

    # ------------------------------------------------------------------ #
    # solve
    # ------------------------------------------------------------------ #
    def _symmetric_layer_specs(self, kxv, kyv, ox, oy, kx0, ky0, k0):
        """Even-parity cascade layer specs (audit F2 for the stack).

        Each layer becomes ``('uniform', eps0)`` or ``('PQ', P, Q, EpsF)`` in
        the SAME projected-operator basis the full solve builds its modes from
        (INTERNAL conj gauge); :func:`_symmetric_cascade_rt` recenters and folds
        them into the ``(Nf+1)``-d even sector.  Reuses ``self._geom_cache``.
        Only scalar / uniform layers reach here (the caller gates out tensors).
        The raw (un-recentered) P/Q go out -- the cascade conjugates them by the
        recentering gauge itself (``GxF/GyF`` are diagonal -> gauge-invariant).
        """
        from .twod import _build_axis, _scalar_projected_ops
        specs, depths = [], []
        for L in self._layers:
            depths.append(L["t"])
            if L["kind"] == "uniform":
                specs.append(("uniform", np.conj(_C(L["eps"]))))
                continue
            tile_i = np.conj(L["tile"])
            eps0 = tile_i.flat[0]
            if bool(np.all(np.abs(tile_i - eps0) < 1e-12)):    # uniform tile
                specs.append(("uniform", eps0))
                continue
            gkey = (L["kind"], L["tile"].tobytes(), L["tile"].shape,
                    tuple(np.ravel(L["xw"])), tuple(np.ravel(L["yw"])),
                    tuple(np.ravel(L["el_x"])), tuple(np.ravel(L["el_y"])))
            gc = self._geom_cache.get(gkey)
            if gc is not None:
                ax, ay, lops = gc
            else:
                ax = _build_axis(self.period_x, L["xw"], self.degree,
                                 L["el_x"], self.grade)
                ay = _build_axis(self.period_y, L["yw"], self.degree,
                                 L["el_y"], self.grade)
                lops = _scalar_projected_ops(ax, ay, tile_i, ox, oy,
                                             self.period_x, self.period_y)
                self._geom_cache[gkey] = (ax, ay, lops)
            GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
            GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
            EpsF, EinvF, EpnF = lops["EpsF"], lops["EinvF"], lops["EpnF"]
            Nf = GxF.shape[0]
            Imat = np.eye(Nf, dtype=_C)
            if self.formulation == "li":
                EPS_nx, EPS_ny, EPS_inv = EpnF, EpsF, EinvF
            else:
                EPS_nx = EPS_ny = EpsF
                EPS_inv = np.linalg.inv(EpsF)
            Q = np.block([[GxF @ GyF, EPS_ny - GxF @ GxF],
                          [GyF @ GyF - EPS_nx, -GyF @ GxF]])
            P = np.block([[GxF @ EPS_inv @ GyF, Imat - GxF @ EPS_inv @ GxF],
                          [GyF @ EPS_inv @ GyF - Imat, -GyF @ EPS_inv @ GxF]])
            specs.append(("PQ", P, Q, EpsF))
        return specs, depths

    def solve(self, *, retain_internal=False):
        """Solve the cascade.  Returns ``(orders, R(2, N), T(2, N),
        jones_reflection(2, 2))`` -- row 0 = incident ``E_x``, row 1 =
        incident ``E_y``; Jones in the PUBLIC convention.

        DIFFERENTIABLE (JAX): any traced input (a uniform-layer eps, a layer
        thickness, the wavelength, ``theta``/``phi``, a half-space index, or
        a traced ``eps_cell`` added with ``region_layout=``) routes to the
        jnp twin (``pmm/_jax_stack2d.py``) -- the SCALAR in-plane vertical
        surface, forward-identical to NumPy at ~1e-15 with AD-vs-FD
        validated gradients.  Tensor layers and ``retain_internal`` raise
        under JAX; x64 required; the eig is CPU-only."""
        # Invalidate retained internals BEFORE any dispatch/early return
        # (audit P1-04): every solve() supersedes the retained state, so
        # internal_field/layer_absorption can only serve the LAST solve --
        # a retain_internal=True one.  Stale fields from a previous
        # source/geometry must never be served silently.  Same contract for
        # the retained per-order amplitudes.
        self._internal = None
        self._modal = None
        # audit B5: the layer-index-keyed Ez-reconstruction cache is tied to the
        # solved state -- clear it here (the single choke point every solve path,
        # incl. the per-wavelength dispersive solve_vs_wavelength loop, passes
        # through) so a re-solve after add_layer / a dispersive layer swap can
        # never serve a stale projected-eps for a reused index.
        self._epsF_cache = {}
        if any(L.get("kind") == "disp" for L in self._layers):
            raise ValueError(
                "PMM2DStackHybrid.solve: the stack holds DISPERSIVE (wl -> value) "
                "materials; use solve_vs_wavelength(wavelengths), which "
                "materialises every callable per wavelength.")
        if self._src is None:
            raise ValueError("PMM2DStackHybrid.solve: call set_source(...) first")
        if not self._layers:
            raise ValueError("PMM2DStackHybrid.solve: add at least one layer")
        wavelength = self._src["wavelength"]
        theta, phi = self._src["theta"], self._src["phi"]

        # ---- differentiable (JAX) dispatch ---------------------------------
        # Any traced input (a uniform-layer eps, a traced eps_cell, a layer
        # thickness, the wavelength, theta/phi or a half-space index) routes
        # to the jnp twin -- the SCALAR in-plane vertical surface.  Tensor
        # layers (and traced tensor cells) raise: the generalized cascade and
        # the tensor assembly are NumPy-only.
        _jax_in = (is_jax_array(self.n_sub) or is_jax_array(self.n_sup)
                   or is_jax_array(wavelength) or is_jax_array(theta)
                   or is_jax_array(phi)
                   or any(is_jax_array(L.get("t")) for L in self._layers)
                   or any(is_jax_array(L.get("eps")) for L in self._layers)
                   or any(L.get("traced") for L in self._layers))
        if _jax_in:
            if retain_internal:
                raise NotImplementedError(
                    "PMM2DStackHybrid.solve(retain_internal=True): not available "
                    "on the JAX (differentiable) path; use NumPy inputs for "
                    "layer_absorption.")
            if any(L["kind"] == "tensor" for L in self._layers):
                raise NotImplementedError(
                    "PMM2DStackHybrid: tensor layers are not differentiable (the "
                    "2-D JAX surface is scalar in-plane); use NumPy inputs "
                    "for tensor stacks.")
            from ._jax_stack2d import _pmm_stack2d_solve_jax
            return _pmm_stack2d_solve_jax(self)

        eps_sup = np.conj(_C(self.n_sup) ** 2)
        eps_sub = np.conj(_C(self.n_sub) ** 2)
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("PMM2DStackHybrid.solve", eps_sup,
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

        # per-layer modes in the shared Rayleigh basis.
        # F4 (audit): dedup repeated identical patterned layers.  Every layer in
        # THIS solve shares (k0, kx0, ky0), so two layers with identical
        # geometry have identical transverse modes; the eig-heavy build runs
        # once and both reuse it (RCWAStack._layer_eig_key pattern).  Modes are
        # thickness-independent (thickness enters the propagation S-matrix), so
        # each layer still carries its OWN thickness.  Bit-exact.
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        orders2d = np.stack([order_x, order_y], axis=1)
        # F2 (audit): even-parity fold of the whole cascade -- opt-in, normal
        # incidence, centro-symmetric scalar/uniform stack (no out-of-plane
        # tensor).  Runs the Redheffer recursion in the (Nf+1)-d EVEN sector;
        # a per-layer flip-invariance guard returns None -> the full solve.
        sym_pairs = None
        if (self.symmetry and not retain_internal
                and abs(kx0) < 1e-12 and abs(ky0) < 1e-12
                and all(L["kind"] != "tensor" for L in self._layers)):
            from ..rcwa._core import _symmetric_cascade_rt
            _specs, _depths = self._symmetric_layer_specs(
                kxv, kyv, ox, oy, kx0, ky0, k0)
            sym_pairs = _symmetric_cascade_rt(
                Vsup, Vsub, np.diag(kxv.astype(_C)), np.diag(kyv.astype(_C)),
                _specs, _depths, k0,
                [np.concatenate([1.0 * delta, 0.0 * delta]),
                 np.concatenate([0.0 * delta, 1.0 * delta])],
                orders2d, np)
        S11 = S21 = None
        if sym_pairs is None:
            modes = []
            _mode_cache = {}
            for L in self._layers:
                if L["kind"] == "uniform":
                    Wl, Vl, lam, _ = _homogeneous_modes(kxv, kyv,
                                                        np.conj(_C(L["eps"])))
                    modes.append(("sym", Wl, Vl, lam, L["t"]))
                    continue
                tile_i = np.conj(L["tile"])
                if L["kind"] == "scalar":
                    eps0 = tile_i.flat[0]
                    if bool(np.all(np.abs(tile_i - eps0) < 1e-12)):  # uniform tile
                        Wl, Vl, lam, _ = _homogeneous_modes(kxv, kyv, eps0)
                        modes.append(("sym", Wl, Vl, lam, L["t"]))
                        continue
                # patterned scalar or tensor -> expensive build + eig; dedup it
                gkey = (L["kind"], L["tile"].tobytes(), L["tile"].shape,
                        tuple(np.ravel(L["xw"])), tuple(np.ravel(L["yw"])),
                        tuple(np.ravel(L["el_x"])), tuple(np.ravel(L["el_y"])))
                hit = _mode_cache.get(gkey)
                if hit is not None:
                    modes.append(hit + (L["t"],))
                    continue
                # F4 part 2: reuse the wl-independent (ax, ay, scalar lops) build
                # across the sweep; only the eig below re-runs per wavelength.
                gc = self._geom_cache.get(gkey)
                if gc is not None:
                    ax, ay, lops = gc
                else:
                    ax = _build_axis(self.period_x, L["xw"], self.degree,
                                     L["el_x"], self.grade)
                    ay = _build_axis(self.period_y, L["yw"], self.degree,
                                     L["el_y"], self.grade)
                    lops = (_scalar_projected_ops(ax, ay, tile_i, ox, oy,
                                                  self.period_x, self.period_y)
                            if L["kind"] == "scalar" else None)
                    self._geom_cache[gkey] = (ax, ay, lops)
                if L["kind"] == "scalar":
                    GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
                    GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
                    Wl, Vl, lam = _layer_modes_projected(
                        GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
                        formulation=self.formulation)
                    entry = ("sym", Wl, Vl, lam)
                else:                                   # tensor (full 3x3)
                    ez_rule = ("li" if self.formulation == "li" else "laurent")
                    out = _tensor_layer_modes(
                        ax, ay, L["xw"], L["yw"], tile_i, k0, kx0, ky0, ox, oy,
                        kxv, kyv, ez_rule)
                    entry = (("gen",) + out if len(out) == 6
                             else ("sym",) + tuple(out))
                _mode_cache[gkey] = entry
                modes.append(entry + (L["t"],))

            any_oop = any(m[0] == "gen" for m in modes)
            if retain_internal and any_oop:
                raise NotImplementedError(
                    "PMM2DStackHybrid.solve(retain_internal=True): symmetric "
                    "(in-plane, vertical) cascades only.")
            if not any_oop:
                # Redheffer cascade: sup | L1 | L2 | ... | Ln | sub (symmetric)
                nlay = len(modes)
                ifc = [_interface_smatrix(Wsup, Vsup, modes[0][1], modes[0][2])]
                for ii in range(1, nlay):
                    ifc.append(_interface_smatrix(
                        modes[ii - 1][1], modes[ii - 1][2],
                        modes[ii][1], modes[ii][2]))
                ifc.append(_interface_smatrix(modes[-1][1], modes[-1][2],
                                              Wsub, Vsub))
                S = ifc[0]
                for ii, (_k, Wl, Vl, lam, t) in enumerate(modes):
                    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * t))
                    S = _redheffer_star(S, ifc[ii + 1])
                if retain_internal:
                    # partial cascades for internal-amplitude recovery (backlog
                    # B1, 2026-06-10 -- the 1-D PMMStack / RCWAStack pattern):
                    # S_above[i] = sup -> TOP of layer i (before its own
                    # propagation); S_below_bot[i] = BOTTOM of layer i -> sub
                    # (without its propagation), so backward amplitudes
                    # reference the layer BOTTOM and both exponentials decay.
                    S_above = [None] * nlay
                    S_above[0] = ifc[0]
                    for ii in range(1, nlay):
                        S_above[ii] = _redheffer_star(
                            _redheffer_star(S_above[ii - 1],
                                            _propagation_smatrix(
                                                modes[ii - 1][3],
                                                k0 * modes[ii - 1][4])),
                            ifc[ii])
                    S_below_bot = [None] * nlay
                    for ii in range(nlay - 1, -1, -1):
                        below = ifc[ii + 1]
                        for jj in range(ii + 1, nlay):
                            below = _redheffer_star(below, _propagation_smatrix(
                                modes[jj][3], k0 * modes[jj][4]))
                            below = _redheffer_star(below, ifc[jj + 1])
                        S_below_bot[ii] = below
                    self._internal = dict(modes=modes, S_above=S_above,
                                          S_below_bot=S_below_bot, k0=k0, Nf=Nf,
                                          kxv=kxv, kyv=kyv, kx0=kx0, ky0=ky0,
                                          order_x=order_x, order_y=order_y,
                                          wl=wl)
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

        # Jones far field (both incident polarizations).  The even-parity
        # cascade (F2) hands back (r, t) per source directly; otherwise the
        # full-basis S-matrix drives each unit polarization.
        kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
        kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
        kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
        safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
        safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
        R_rows, T_rows, j_cols = [], [], []
        amp = {k: np.zeros((2, Nf), dtype=_C)
               for k in ("rx", "ry", "tx", "ty")}
        for ip, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            long_inc = (kx0 * ex0 + ky0 * ey0)
            einc_sq = (1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0
                       else 1.0)
            if sym_pairs is not None:
                r, t_ = sym_pairs[ip]
            else:
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
            # PUBLIC exp(-iwt) amplitudes = conj of this INTERNAL-gauge
            # cascade (the same map the Jones takes above).
            amp["rx"][ip], amp["ry"][ip] = np.conj(rx), np.conj(ry)
            amp["tx"][ip], amp["ty"][ip] = np.conj(tx), np.conj(ty)
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jones = np.stack(j_cols, axis=1)
        # AUDIT_DYNAMETA_CONSUMER_API_GAPS B: retained per-order amplitudes
        # (PerOrderAmplitudesMixin contract).  kz_ref_f/kz_trn_f are computed
        # on conj(eps_internal) = the PUBLIC eps -> public decaying branch.
        self._modal = dict(
            orders=orders2d.copy(), p0=p0,
            kx=kxv.copy(), ky=kyv.copy(),
            kz_ref=kz_ref_f.copy(), kz_trn=kz_trn_f.copy(),
            kz_inc=kz_inc, kx0=float(kx0), ky0=float(ky0),
            wavelength=float(wl), **amp)
        if retain_internal and getattr(self, "_internal", None) is not None:
            cinc_cols = [np.concatenate([1.0 * delta, 0.0 * delta]),
                         np.concatenate([0.0 * delta, 1.0 * delta])]
            self._internal["cinc"] = np.stack(cinc_cols, axis=1).astype(_C)
            self._internal["R_tot"] = R_eff.sum(axis=1)
            self._internal["T_tot"] = T_eff.sum(axis=1)
        _warn_stack_energy(R_eff, T_eff)
        return orders2d, R_eff, T_eff, jones

    def _layer_epsF(self, i):
        """Projected ``[[ezz]]`` of layer ``i`` in the shared Fourier basis
        (INTERNAL conj gauge) -- the ``Ez`` reconstruction operator, the
        exact mirror of the RCWA route ``solve(EPS, -(Kx Hy - Ky Hx))``.
        ``None`` for a uniform(-valued) layer (diagonal exact path).

        NB the hybrid 2-D PMM solution LIVES in the projected Fourier basis
        (the nodal grid is the operators' quadrature device, not a field
        representation), so Fourier-side reconstruction is the consistent
        one; pushing ``n_orders`` toward the axis nodal capacity degrades
        the highest projected orders (pinv conditioning) and with them
        ``Ez`` -- keep ``n_orders`` comfortably inside the validated
        capacity (the solver's usual regime), as for the solve itself."""
        cache = getattr(self, "_epsF_cache", None)
        if cache is None:
            cache = self._epsF_cache = {}
        if i in cache:
            return cache[i]
        from .twod import _build_axis, _scalar_projected_ops
        L = self._layers[i]
        out = None
        if L["kind"] in ("scalar", "tensor"):
            # only ezz enters the Ez constitutive reconstruction, so an
            # IN-PLANE tensor layer projects its zz component the same way
            tile_i = (np.conj(L["tile"]) if L["kind"] == "scalar"
                      else np.conj(L["tile"][..., 2, 2]))
            eps0 = tile_i.flat[0]
            if not bool(np.all(np.abs(tile_i - eps0) < 1e-12)):
                ax = _build_axis(self.period_x, L["xw"], self.degree,
                                 L["el_x"], self.grade)
                ay = _build_axis(self.period_y, L["yw"], self.degree,
                                 L["el_y"], self.grade)
                ox = np.arange(-self.n_orders, self.n_orders + 1)
                oy = np.arange(-self.n_orders, self.n_orders + 1)
                lops = _scalar_projected_ops(ax, ay, tile_i, ox, oy,
                                             self.period_x, self.period_y)
                out = lops["EpsF"]
        cache[i] = out
        return out

    def internal_field(self, z, *, component="all", nx=64, ny=None, dx=None,
                       dy=None, layer=None, incident=(1.0, 0.0),
                       filter="none"):
        """Reconstruct the real-space E and/or H field INSIDE the 2-D stack --
        the crossed-grating mirror of :meth:`RCWAResult.internal_field`
        (same grid conventions, carriers and output layout, so the two
        co-register pointwise).

        Requires that the MOST RECENT ``solve`` used
        ``retain_internal=True`` (symmetric -- i.e. in-plane vertical --
        cascades); any re-solve invalidates previously retained internals.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] from the stack TOP; with ``layer=i`` given, ``z`` is
            LOCAL to layer ``i``.
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default all six).
        nx, ny : int, optional
            Real-space grid (``ny`` defaults to ``nx``).
        dx, dy : float, optional
            Grid pitch [m] (default tiles one unit cell).
        layer : int, optional
            Force a layer index (``z`` then local).
        incident : (complex, complex), optional
            PUBLIC incident Jones ``(E_x, E_y)`` (default x-polarized;
            complex values are handled in the public gauge -- the
            handedness-correct superposition).
        filter : {'none', 'lanczos'}, optional
            Lanczos damping of the high orders (Gibbs suppression).

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``), each
            ``(nz, ny, nx)`` complex (``ny`` axis dropped when 1), plus
            ``'z'``, ``'x'``, ``'y'``.  PUBLIC ``exp(-i w t)`` convention,
            ``H`` in the RCWA-co-registered ``-i eta0`` scale.
        """
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMM2DStackHybrid.internal_field: no internal data retained; the "
                "MOST RECENT solve must use solve(retain_internal=True) "
                "(any re-solve invalidates previously retained internals).")
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"PMM2DStackHybrid.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        if filter not in ("none", "lanczos"):
            raise ValueError(
                f"PMM2DStackHybrid.internal_field: filter must be 'none' or "
                f"'lanczos', got {filter!r}.")
        want = names[component]
        Nf = d["Nf"]
        k0 = d["k0"]
        kxv, kyv = d["kxv"], d["kyv"]
        order_x, order_y = d["order_x"], d["order_y"]
        thick = [L["t"] for L in self._layers]
        z_top = np.concatenate([[0.0], np.cumsum(thick)])

        # PUBLIC incident -> INTERNAL (conj) gauge superposition weights (the
        # cascade runs conjugated and the output is conjugated back).
        amps = self._internal_amplitudes()
        wx, wy = np.conj(incident[0]), np.conj(incident[1])

        ny_i = int(nx) if ny is None else int(ny)
        nx_i = int(nx)
        dx_f = float(self.period_x / nx_i if dx is None else dx)
        dy_f = float(self.period_y / ny_i if dy is None else dy)
        xg = (np.arange(nx_i) - nx_i // 2) * dx_f
        yg = (np.arange(ny_i) - ny_i // 2) * dy_f
        X, Y = np.meshgrid(xg, yg)                          # (ny, nx)
        carriers = np.exp(1j * k0 * (np.real(kxv)[:, None, None] * X[None]
                                     + np.real(kyv)[:, None, None] * Y[None]))
        if filter == "lanczos":
            mx = max(1, int(np.abs(order_x).max()))
            my = max(1, int(np.abs(order_y).max()))
            sigma = (np.sinc(order_x / (mx + 1.0))
                     * np.sinc(order_y / (my + 1.0)))
        else:
            sigma = None

        Kx = kxv.astype(_C)
        Ky = kyv.astype(_C)
        zs = np.atleast_1d(np.asarray(z, dtype=float))
        out = {c: np.empty((zs.shape[0], ny_i, nx_i), dtype=_C) for c in want}
        cache = {}
        for zi, zz in enumerate(zs):
            if layer is None:
                i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                                0, len(thick) - 1))
                zloc = zz - z_top[i]
            else:
                i, zloc = int(layer), float(zz)
            _k, Wl, Vl, lam, t_i = d["modes"][i]
            if i not in cache:
                c_fwd, c_bwd = amps[i]
                cache[i] = (wx * c_fwd[:, 0] + wy * c_fwd[:, 1],
                            wx * c_bwd[:, 0] + wy * c_bwd[:, 1])
            cF, cB = cache[i]
            P = np.exp(-lam * k0 * zloc)
            Q = np.exp(-lam * k0 * max(t_i - zloc, 0.0))
            sv = Wl @ (P * cF) + Wl @ (Q * cB)
            uv = Vl @ (P * cF) - Vl @ (Q * cB)
            Ex_m, Ey_m = sv[:Nf], sv[Nf:]
            Hx_m, Hy_m = -1j * uv[:Nf], -1j * uv[Nf:]
            h = dict(Ex=Ex_m, Ey=Ey_m, Hx=Hx_m, Hy=Hy_m)
            if "Ez" in want:
                EpsF = self._layer_epsF(i)
                if EpsF is None:
                    Li_ = self._layers[i]
                    if Li_["kind"] == "uniform":
                        eps_l = np.conj(_C(Li_["eps"]))
                    elif Li_["kind"] == "scalar":
                        eps_l = np.conj(Li_["tile"].flat[0])
                    else:
                        eps_l = np.conj(Li_["tile"][..., 2, 2].flat[0])
                    h["Ez"] = -(Kx * Hy_m - Ky * Hx_m) / eps_l
                else:
                    h["Ez"] = np.linalg.solve(EpsF,
                                              -(Kx * Hy_m - Ky * Hx_m))
            if "Hz" in want:
                h["Hz"] = Kx * Ey_m - Ky * Ex_m
            for c in want:
                amp = np.conj(h[c])                         # internal -> public
                if sigma is not None:
                    amp = amp * sigma
                out[c][zi] = np.tensordot(amp, carriers, axes=([0], [0]))
        result = {c: (out[c][:, 0, :] if ny_i == 1 else out[c]) for c in want}
        result["z"] = zs
        result["x"] = xg
        result["y"] = yg
        return result

    def _internal_amplitudes(self):
        """Per-layer ``(c_fwd_top, c_bwd_bot)`` modal amplitudes from the
        retained partial cascades (both incident polarizations)."""
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMM2DStackHybrid: no internal data retained; the MOST RECENT "
                "solve must use solve(retain_internal=True) (any re-solve "
                "invalidates previously retained internals).")
        out = []
        k0 = d["k0"]
        for i, (_k, _Wl, _Vl, lam, t) in enumerate(d["modes"]):
            Sa = d["S_above"][i]
            Sb = d["S_below_bot"][i]
            X = np.exp(-lam * k0 * t)
            n = lam.shape[0]
            A22, A21 = Sa[3], Sa[2]
            B11 = Sb[0]
            M = np.eye(n, dtype=_C) - (A22 * X[None, :]) @ (B11 * X[None, :])
            c_fwd = np.linalg.solve(M, A21 @ d["cinc"])
            c_bwd = B11 @ (X[:, None] * c_fwd)
            out.append((c_fwd, c_bwd))
        return out

    def _flux_at(self, i, z_frac, amps):
        """z-Poynting flux through a plane inside layer ``i`` in the shared
        Rayleigh basis (Parseval: the order basis is orthogonal, so the
        cell-integrated flux is the plain coefficient sum with the same
        modal-H ``-i`` convention the 1-D machinery uses)."""
        d = self._internal
        _k, Wl, Vl, lam, t = d["modes"][i]
        c_fwd, c_bwd = amps[i]
        k0, Nf = d["k0"], d["Nf"]
        P = np.exp(-lam * k0 * (z_frac * t))[:, None]
        Q = np.exp(-lam * k0 * ((1.0 - z_frac) * t))[:, None]
        E = Wl @ (P * c_fwd) + Wl @ (Q * c_bwd)
        H = Vl @ (P * c_fwd) - Vl @ (Q * c_bwd)
        Ex, Ey = E[:Nf], E[Nf:]
        Hx, Hy = H[:Nf], H[Nf:]
        return np.array([
            float(np.imag(np.sum(Ex[:, c] * np.conj(Hy[:, c])
                                 - Ey[:, c] * np.conj(Hx[:, c]))))
            for c in range(2)])

    def layer_absorption(self):
        """Per-layer absorbed power fraction ``(n_layers, 2)`` (backlog B1,
        2026-06-10) -- the 2-D mirror of :meth:`PMMStack.layer_absorption`,
        from the internal z-Poynting flux difference across each layer.
        Requires that the MOST RECENT ``solve`` used
        ``retain_internal=True`` (symmetric cascades; any re-solve
        invalidates previously retained internals).  The
        cross-machinery invariant ``sum_i A_i ~= 1 - sum R - sum T`` is the
        honest closure check (internal amplitudes vs Rayleigh far field).

        Per-MATERIAL attribution is not offered in 2-D yet: the retained
        modes live in the projected Rayleigh basis (the nodal volume data is
        not stored) -- a follow-up if an application needs the split."""
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMM2DStackHybrid.layer_absorption: no internal data retained; "
                "the MOST RECENT solve must use solve(retain_internal=True) "
                "(any re-solve invalidates previously retained internals).")
        amps = self._internal_amplitudes()
        nlay = len(d["modes"])
        F_top = np.array([self._flux_at(i, 0.0, amps) for i in range(nlay)])
        F_bot = np.array([self._flux_at(i, 1.0, amps) for i in range(nlay)])
        one_minus_R = 1.0 - d["R_tot"]
        F_inc = np.where(np.abs(one_minus_R) > 1e-15,
                         F_top[0] / one_minus_R, np.inf)
        return (F_top - F_bot) / F_inc[None, :]

    def _materialized_layers(self, w, layers):
        """Concrete layer list at one wavelength: each DISPERSIVE layer's
        callable is resolved and run through the normal add-time processing
        (walls/tile extraction + validation); concrete layers pass through."""
        out = []
        for L in layers:
            if L.get("kind") != "disp":
                out.append(L)
                continue
            probe = PMM2DStackHybrid.__new__(PMM2DStackHybrid)
            probe.__dict__.update(self.__dict__)
            probe._layers = []
            probe.add_layer(L["t"], **{L["slot"]: L["fn"](float(w))})
            out.extend(probe._layers)
        return out

    def solve_vs_wavelength(self, wavelengths, *, theta=None, phi=None,
                            angle=None, jones=False, max_workers=None,
                            blas_per_worker=1):
        """Solve the stack across a wavelength sweep, materialising any
        DISPERSIVE (``wl -> value``) layer callables per wavelength (item 5,
        device-geometry roadmap 2026-06-10).
        Returns ``(orders, R(n_wl, 2, N), T(n_wl, 2, N))``, plus a FOURTH
        ``jones (n_wl, 2, 2)`` element when ``jones=True`` (default ``False``
        keeps the released 3-tuple).

        v5.20 (audit B3): when neither ``theta``/``angle`` nor ``phi`` is
        passed, the sweep REUSES a previously ``set_source()``-configured
        incidence instead of silently resetting to normal incidence.  Any
        explicit ``theta``/``angle``/``phi`` still wins.

        The per-wavelength solves are INDEPENDENT (each on a private shallow
        clone) and release the GIL inside LAPACK, so they run on a bounded thread
        pool (``max_workers=None`` picks ``min(cpu, n_wl)``; each worker pins its
        BLAS pool to ``blas_per_worker``).  Results stored BY INDEX ->
        BYTE-IDENTICAL to serial for any worker count (``max_workers=1`` forces
        serial).  Each per-wavelength ``solve()`` already dedups identical layers."""
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        # audit B3: inherit the configured source angle/azimuth when the sweep
        # leaves them unset (theta/angle both None -> reuse _src['theta'];
        # phi None -> reuse _src['phi']).  Falls back to normal incidence when
        # no set_source() was called (the pre-B3 default).
        src = getattr(self, "_src", None)
        if theta is None and angle is None and src is not None:
            theta = src.get("theta")
        if phi is None:
            phi = src.get("phi", 0.0) if src is not None else 0.0
        wlv = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        if wlv.size == 0:
            raise ValueError("PMM2DStackHybrid.solve_vs_wavelength: wavelengths is "
                             "empty; pass at least one wavelength [m].")
        if not np.all(np.isfinite(wlv)) or np.any(wlv <= 0.0):
            raise ValueError("PMM2DStackHybrid.solve_vs_wavelength: every "
                             "wavelength must be a finite value > 0 [m].")
        base = self._layers

        def _solve_one(i, w):
            # Independent solve on a PRIVATE shallow clone (own _layers/_src/
            # _internal; the module-level geo-eig cache is lock-guarded, and
            # solve() dedups identical layers), so no worker mutates shared self.
            w = float(w)
            sub = _copy.copy(self)
            sub._internal = None
            sub._layers = self._materialized_layers(w, base)
            sub.set_source(w, theta=theta, phi=phi, angle=angle)
            with _blas_threads_quiet(blas_per_worker):
                o, R1, T1, j1 = sub.solve()
            return i, o, R1, np.asarray(T1), np.asarray(j1)

        results = [None] * wlv.size
        # A traced (JAX) half-space forces serial (concurrent tracing is not
        # thread-safe); a traced LAYER cell makes solve() return traced arrays
        # that np.stack rejects cleanly below, exactly as the pre-thread loop did.
        _jax_any = is_jax_array(self.n_sup) or is_jax_array(self.n_sub)
        _mw = (1 if _jax_any else
               (min(os.cpu_count() or 1, int(wlv.size)) if max_workers is None
                else max(1, int(max_workers))))
        if _mw == 1 or wlv.size == 1:
            for i, w in enumerate(wlv):
                results[i] = _solve_one(i, w)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=_mw) as _ex:
                for res in _ex.map(lambda p: _solve_one(*p),
                                   list(enumerate(wlv))):
                    results[res[0]] = res
        orders = results[0][1]
        R = np.stack([r[2] for r in results])
        T = np.stack([r[3] for r in results])
        J = np.stack([r[4] for r in results])
        if jones:
            return orders, R, T, J
        return orders, R, T


# ============================ NAME MIGRATION ============================== #
# The class was renamed PMM2DStack -> PMM2DStackHybrid to make the HYBRID
# (Fourier-Galerkin projected -> FMM n_orders floor) approach explicit, freeing
# the plain ``PMM2DStack`` name for the no-floor PURE staggered stack
# (:class:`~lumenairy.elements.pmm.stack2d_pure.PMM2DStackPure`).
#
# ``PMM2DStack_hybrid`` is a permanent explicit alias.  ``PMM2DStack`` is a
# TRANSITIONAL alias that currently still resolves to the hybrid so existing
# code keeps working; it will be repointed to the pure stack once that reaches
# feature + validation parity (a deliberate, tested cutover).  Import
# ``PMM2DStackHybrid`` (or ``PMM2DStackPure``) explicitly to pin the method.
PMM2DStack_hybrid = PMM2DStackHybrid


class PMM2DStack(PMM2DStackHybrid):
    """TRANSITIONAL alias for :class:`PMM2DStackHybrid` -- emits a
    ``DeprecationWarning`` on construction (D9).

    The bare ``PMM2DStack`` name currently resolves to the HYBRID
    (Fourier-projected, FMM ``n_orders``-floor) stack, but is scheduled to be
    REPOINTED to the no-floor pure staggered stack
    (:class:`~lumenairy.elements.pmm.stack2d_pure.PMM2DStackPure`) once that
    reaches feature + validation parity -- a SILENT results change for callers
    of the bare name (Fourier floor -> no floor; different convergence knobs).
    The deprecation phase warns now so code can pin the method BEFORE the
    cutover.  Import :class:`PMM2DStackHybrid` to keep today's behaviour, or
    :class:`PMM2DStackPure` for the pure method.  ``isinstance`` against
    ``PMM2DStackHybrid`` still holds (this is a subclass, adding no behaviour).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PMM2DStack is a TRANSITIONAL alias for PMM2DStackHybrid and is "
            "scheduled to be repointed to the no-floor pure stack "
            "(PMM2DStackPure) in a future release -- a silent results change "
            "(Fourier floor -> no floor, different convergence knobs).  Import "
            "PMM2DStackHybrid to keep the current behaviour, or PMM2DStackPure "
            "for the pure method.",
            DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
