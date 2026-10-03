"""
lumenairy.elements.pmm._jax_twod_staggered -- JAX twin of the pure staggered 2-D PMM.
=====================================================================================

Phase E3 of the curved-cell plan (``docs/audits/PLAN_PMM2D_CURVED_CELLS_
2026_09_26.md`` section 4.5; build record ``docs/audits/BUILD_PMM2D_CURVED_E3_
2026_10_03.md``): a differentiable forward solve of the SHARED-GRID
:class:`~lumenairy.elements.pmm.PMM2DStackPure` cascade -- scalar, block-form
tensor and magnetic cells, with and without a curved coordinate map -- whose
gradients flow to the layer permittivities and permeabilities (real and
imaginary parts), the thicknesses, the half-space indices, and the SHAPE
PARAMETERS of the map (a circle's radius, a fillet's radius, a sinusoid's
amplitude, a rectangle's width).

What this module holds, and what it does NOT
--------------------------------------------
The ONE-kernel rule in its JAX form: the twin runs the NumPy module's own
functions with ``xp = jax.numpy`` wherever they are array-module generic --
the assembly :meth:`~lumenairy.elements.pmm.twod_staggered.Granet2DTransverseE._assemble`
(on a SHADOW copy of a NumPy-built solver), the quadrature kernel
:func:`~lumenairy.elements.pmm.twod_staggered._stag_quad_weighted`, the
effective-tensor kernels ``_stag_map_eff`` / ``_stag_map_eff_tensor``, the
region-mode post-processing ``_region_modes_from_eig`` /
``_homog_geom_from_eig`` / ``_homog_region_modes``, the cofactor far field
``_far_projector_mapped``, the exact incident decomposition
``_stag_incident_coeffs_mapped``, the order projection, the per-order ``kz``
and the efficiency projection, and the library's Redheffer algebra
(``rcwa._core``).  The census test pins every one of them as the NumPy
module's object.  Only what cannot be shared lives here: the differentiable
pencil eig (:func:`_stag_geneig_jax`, through ``rcwa._jax_eig_stable``), the
frozen template, and the cascade driver.

The frozen DISCRETE decisions
-----------------------------
A JAX trace cannot branch on a value.  Everything the NumPy solve decides
from values is decided ONCE, on the concrete REFERENCE geometry the stack was
built with, and frozen into the template:

* the ``(u, v)`` wall grid and the cell topology (which cell holds which
  material, which edges are curves, which corners are singular vertices);
* the adaptive quadrature node count of the mapped assembly
  (``_stag_map_nodes``) and the corner (Duffy) cells;
* the per-cell node counts of the far-field projector and of the incident
  load (sized from each cell's physical extent);
* the Wood-anomaly wavelength nudge and the propagating-incidence check
  (the wavelength and the angles are STATIC in this twin: they set the Bloch
  glue of the basis);
* the stack's eig de-duplication.

A traced shape parameter then moves the IMAGES of the frozen grid (vertex
images and edge curves) -- the map, its analytic Jacobian, the geometric
weights at the frozen nodes -- and nothing else, so the traced function is
smooth in the parameter.  As argued and measured in the build record, the
discrete solution does not depend on WHERE the frozen ``(u, v)`` walls sit,
only on the physical images of the cells, so the twin evaluated at a
parameter ``p`` reproduces the NumPy solve built at ``p`` (whose own grid
moves with ``p``) to round-off at normal incidence and to the level of the
incident decomposition's representation error at oblique incidence.

A parameter path that CHANGES the topology -- a fold (``det J <= 0`` at a
node), a segment crossing the sliver contract, two snapped walls separating,
a fillet radius reaching zero (the singular vertices disappear) -- is a
NON-DIFFERENTIABLE event of the underlying discretisation.  The twin poisons
its outputs with NaN when it detects one in the trace (gate E3-4), and its
host-side check refuses it when the value is concrete.

Scope
-----
In-plane only, shared grid only.  An OUT-OF-PLANE tensor (with or without a
map), ``slant`` (Phase E1 of the plan: out-of-plane and slant under a map)
and ``layer_grids='per-layer'`` (Phase E2: per-layer maps) RAISE, naming the
follow-up.  ``retain_internal`` / ``layer_absorption`` are NumPy-only.  x64 is
required; ``jnp.linalg.eig`` is CPU-only.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..rcwa._core import _norm_slant_pair, _slant_is_zero
from . import twod_staggered as TS
from ._core import _guarded_lstsq
from .twod_jones import _tile_is_offplane

_C = np.complex128

__all__ = ["StagJaxTwin"]

#: The eigenvector-VJP regularisation passed to ``rcwa._jax_eig_stable``
#: (fraction of ``max |lam|`` below which a splitting counts as unresolved).
#: ``None`` = that function's own default (``rcwa._core._EIG_TAU_REL``).
#: Gate E3-5 measures its effect on the degenerate Bloch modes of a
#: four-fold symmetric circle.
_E3_EIG_TAU_REL = None

#: The degenerate-CLUSTER rule of the reverse pass (verifier V-E3-1; round 2
#: of the build record): ``rcwa._jax_eig_cluster_adjoint``'s ``gap_rel`` and
#: ``split_rel``.  ``None`` = that function's defaults
#: (``rcwa._core._EIG_CLUSTER_GAP_REL`` / ``_EIG_CLUSTER_SPLIT_REL``); a
#: ``gap_rel <= 0`` switches the rule off (the plain eig VJP, whose gradient
#: is WRONG for a symmetry-breaking parameter at a symmetric cell).
_E3_EIG_CLUSTER_GAP_REL = None
_E3_EIG_CLUSTER_SPLIT_REL = None


def _stag_geneig_jax(L, G, tau_rel=None):
    """Differentiable eig of the IN-PLANE pencil ``L W = g2 G W`` (``G = -R``,
    Hermitian positive definite in every in-plane region, mapped or not).

    The NumPy path runs QZ (``scipy.linalg.eig(L, G)``); JAX has no
    generalized eig, so the pencil is reduced to the standard eig of
    ``G^-1 L`` (same eigenvalues, same eigenvectors -- ``G`` is invertible)
    and differentiated through the library's ONE gauge-stable custom-VJP eig
    ``rcwa._jax_eig_stable`` (Lorentzian-broadened eigenvector cotangent).
    Every quantity downstream (R, T, the Jones matrix) is invariant under a
    per-mode rescaling of ``W``, so the different eigenvector normalisation
    of the two eigensolvers is invisible in the outputs."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable
    A = jnp.linalg.solve(G, L)
    eig = _jax_eig_stable()
    tau = _E3_EIG_TAU_REL if tau_rel is None else tau_rel
    if tau is None:
        return eig(A)
    return eig(A, tau)


def _shadow(ref, xp, eps_cell, mu_cell, cmap):
    """A SHADOW of the NumPy-built solver ``ref``: the same basis, grid,
    quadrature rule and factor cache (all frozen), with the materials (and the
    map) replaced by possibly-traced values, assembled by ``ref``'s own
    :meth:`_assemble` with ``_xp = xp``."""
    sh = copy.copy(ref)
    sh._xp = xp
    sh.eps_cell = eps_cell
    sh.mu_cell = mu_cell
    if ref.cmap is not None:
        sh.cmap = cmap
        sh._mapw = TS._stag_map_weights(ref.bx, ref.by, cmap, eps_cell,
                                        ref._qrule, mu_cell=mu_cell, xp=xp)
    sh._assemble()
    return sh


def _min_detj(ref, cmap, xp):
    """``min det J`` over every node of the mapped assembly (tensor rule and
    corner rule) -- the in-trace fold guard (gate E3-4)."""
    quad = ref._qrule
    J4, P4 = TS._stag_map_node_jacobian(ref.bx, ref.by, cmap, quad,
                                        quad.tensor[0], xp)
    xu, xv, yu, yv = J4
    sg = [xp.min(xu * yv - xv * yu)]
    for pu, pv, qu, qv in P4.values():
        sg.append(xp.min(pu * qv - pv * qu))
    return xp.min(xp.stack(sg))


def _traced_shape_merge(parts, ref_layers, t_layers, px, py, cmap_ref):
    """REPLAY of :func:`~lumenairy.elements.pmm.shapes2d._merge` with traced
    shape parameters on the reference merge's frozen structure (Phase E3).

    ``parts`` is the reference merge's intermediates (``_merge(...,
    parts=True)``), ``ref_layers`` / ``t_layers`` the reference and the
    (possibly traced) ``(shapes, background_eps, background_mu)`` of every
    shape layer, in the same structure.  Every structural decision -- which
    walls merge, which grid vertex a shape vertex is, which edges are curve
    pieces and of which curve, which cells a shape paints, how the grid was
    squared -- is the reference's; every VALUE is the traced shapes' own,
    computed by the primitives' own ``_layout`` (``ref=`` the concrete shape)
    and the merge's own ``_curve_piece`` / ``_curve_point``.  Returns
    ``(cmap, cells, mu_cells, ok)``: the traced map on the frozen ``(u, v)``
    grid, the per-layer (traced) permittivity / permeability cells, and a
    traced boolean that is False when the parameters left the reference
    topology (two merged walls separating, two claims on one vertex or edge
    diverging, a merged segment below the sliver contract)."""
    import jax.numpy as jnp

    from . import shapes2d as SH
    from ._curvemap import RefinedMap, TransfiniteMap
    scale = max(px, py)
    tol = 1e-9 * scale
    items = parts["items"]
    gu, gv = parts["gu"], parts["gv"]
    flat_t = [sh for shapes, _b, _m in t_layers for sh in shapes]
    if len(flat_t) != len(items):
        raise ValueError("PMM2DStackPure (JAX): the traced shapes do not "
                         "match the reference stack's shapes (count).")
    for (_w, sh_r, _l), sh_t in zip(items, flat_t):
        if type(sh_t) is not type(sh_r) or any(
                getattr(sh_t, k, None) != getattr(sh_r, k, None)
                for k in ("core", "axis", "period_count")) or (
                (getattr(sh_t, "width", None) is None)
                != (getattr(sh_r, "width", None) is None)) or (
                sh_t.is_magnetic != sh_r.is_magnetic) or (
                sh_t.is_tensor != sh_r.is_tensor):
            raise ValueError(
                f"PMM2DStackPure (JAX): the traced shape {sh_t!r} does not "
                f"have the structure of the reference shape {sh_r!r} (type, "
                f"layout option, tensor / magnetic material); the twin's "
                f"topology is frozen at the reference -- rebuild the twin.")
    lays_t = [sh_t._layout(px, py, ref=sh_r)
              for (_w, sh_r, _l), sh_t in zip(items, flat_t)]
    devs = []

    def walls(g, attr):
        vals: list[Any] = [None] * g.b.size
        vals[0], vals[-1] = 0.0, g.period
        for (_w, _s, lay_r), lay_t in zip(items, lays_t):
            for wr, wt in zip(getattr(lay_r, attr), getattr(lay_t, attr)):
                k = g.index(wr)
                if vals[k] is None:
                    vals[k] = wt
                else:
                    devs.append(wt - vals[k])
        return vals
    U1, V1 = walls(gu, "u_walls"), walls(gv, "v_walls")
    vcl: dict[Any, list[Any]] = {}
    ecl: dict[Any, list[Any]] = {}
    for (_w, _s, lay_r), lay_t in zip(items, lays_t):
        ends = {}
        vt = lay_t.vertices
        vt = list(vt.items()) if isinstance(vt, dict) else vt
        for ((u, v), _xy), (_k, xy_t) in zip(lay_r.vertices.items(), vt):
            key = (gu.index(u), gv.index(v))
            ends[key] = xy_t
            vcl.setdefault(key, []).append(xy_t)
        for e_r, e_t in zip(lay_r.edges, lay_t.edges):
            if e_r.kind == "h":
                j = gv.index(e_r.fixed)
                ia, ib = gu.index(e_r.a), gu.index(e_r.b)
                run_r, run_t = gu.b, U1
                pa, pb = ends[(ia, j)], ends[(ib, j)]
            else:
                i = gu.index(e_r.fixed)
                ia, ib = gv.index(e_r.a), gv.index(e_r.b)
                run_r, run_t = gv.b, V1
                pa, pb = ends[(i, ia)], ends[(i, ib)]
            for k in range(ia, ib):
                ekey = ("h", k, j) if e_r.kind == "h" else ("v", i, k)
                ta_r = e_r.a if k == ia else float(run_r[k])
                tb_r = e_r.b if k + 1 == ib else float(run_r[k + 1])
                ta_t = e_t.a if k == ia else run_t[k]
                tb_t = e_t.b if k + 1 == ib else run_t[k + 1]
                ecl.setdefault(ekey, []).append(SH._curve_piece(
                    e_t, ta_t, tb_t, ref=(e_r, ta_r, tb_r)))
            for k in range(ia + 1, ib):
                key = (k, j) if e_r.kind == "h" else (i, k)
                vcl.setdefault(key, []).append(SH._curve_point(
                    e_t, run_t[k], (pa, pb), ref=e_r))
    nx, ny = gu.b.size - 1, gv.b.size - 1
    rows = []
    for i in range(nx + 1):
        col = []
        for j in range(ny + 1):
            cl = vcl.get((i, j))
            if cl is None:
                col.append(jnp.stack([jnp.asarray(U1[i], dtype=jnp.float64),
                                      jnp.asarray(V1[j], dtype=jnp.float64)]))
                continue
            xy0 = jnp.asarray(cl[0], dtype=jnp.float64).reshape(2)
            for xy in cl[1:]:
                devs.append(jnp.max(jnp.abs(jnp.asarray(xy).reshape(2)
                                            - xy0)))
            col.append(xy0)
        rows.append(jnp.stack(col))
    Vimg = jnp.stack(rows)
    curved = {}
    s5 = np.linspace(0.0, 1.0, 5)
    for ek, cl in ecl.items():
        first = next((c for c in cl if c is not None), None)
        if first is None:
            continue
        curved[ek] = first
        ref_s = first(s5)[0]
        for c in cl:
            if c is not None and c is not first:
                devs.append(jnp.max(jnp.abs(c(s5)[0] - ref_s)))
    tm_r = parts["tm"]
    tm = TransfiniteMap._traced(tm_r.u_bounds, tm_r.v_bounds, Vimg, curved,
                                tm_r.singular_vertices)
    cmap = (RefinedMap._traced(tm, cmap_ref)
            if isinstance(cmap_ref, RefinedMap) else tm)
    # topology guard: merged walls ordered and above the sliver contract
    ok = jnp.asarray(True)
    for vals, per in ((U1, px), (V1, py)):
        w = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in vals])
        ok = ok & jnp.all(jnp.diff(w) >= TS._STAG_MIN_SEG_FRAC * per
                          * (1.0 - 1e-9))
    for d in devs:
        ok = ok & (jnp.max(jnp.abs(d)) <= tol)
    # the merge's INSIDE-THE-CELL contracts, traced (a concrete call is
    # refused by the merge; verifier V-E3-3): a sinusoidal wall's
    # base -+ |amplitude| at least the sliver width inside the cell
    # (SinusoidalWall._layout), any other curved shape's bounding box at least
    # the sliver width from the cell edges, a rectangle's inside it
    # (Shape2D._check_inside)
    for (_w, sh_r, _l), sh_t in zip(items, flat_t):
        if isinstance(sh_r, SH.SinusoidalWall):
            pp = px if sh_r.axis == "x" else py
            A = jnp.abs(sh_t.amplitude)
            for base in sh_t._bases():
                ok = ok & (base - A >= TS._STAG_MIN_SEG_FRAC * pp) & (
                    base + A <= pp * (1.0 - TS._STAG_MIN_SEG_FRAC))
            continue
        x0, x1, y0, y1 = sh_t.bbox()
        strict = not isinstance(sh_r, SH.Rect)
        mx = TS._STAG_MIN_SEG_FRAC * px if strict else -1e-12 * px
        my = TS._STAG_MIN_SEG_FRAC * py if strict else -1e-12 * py
        ok = ok & (x0 >= mx) & (y0 >= my) & (x1 <= px - mx) & (y1 <= py - my)
    # painting (the reference's masks, the traced materials)
    uc, vc = parts["uc"], parts["vc"]
    N = uc.size
    cj = jnp.complex128
    cells: list[Any] = []
    mus: list[Any] = []
    for (shp_r, bg_r, bgm_r), (shp_t, bg_t, bgm_t) in zip(ref_layers,
                                                        t_layers):
        tensor = np.ndim(SH._as_eps(bg_r, "bg")) == 2 or any(
            sh.is_tensor for sh in shp_r)

        def paint(base, vals, tensor):
            def as_cell(v):
                a = jnp.asarray(v).astype(cj)
                if tensor:
                    a = a if a.ndim == 2 else a * jnp.eye(3, dtype=cj)
                    return jnp.broadcast_to(a, (N, N, 3, 3))
                return jnp.broadcast_to(a, (N, N))
            cell = as_cell(base)
            for sh_r, v in zip(shp_r, vals):
                m = SH._fill_mask(sh_r._layout(px, py), uc, vc)
                mm = m[:, :, None, None] if tensor else m
                cell = jnp.where(mm, as_cell(v), cell)
            return cell
        cells.append(paint(bg_t, [sh.eps for sh in shp_t], tensor))
        if bgm_r is None and not any(sh.is_magnetic for sh in shp_r):
            mus.append(None)
            continue
        mt = any(np.ndim(m) == 2 for m in
                 [bgm_r if bgm_r is not None else 1.0]
                 + [sh.mu for sh in shp_r if sh.is_magnetic])
        mus.append(paint(bgm_t if bgm_t is not None else 1.0 + 0.0j,
                         [sh.mu if sh.is_magnetic else 1.0 + 0.0j
                          for sh in shp_t], mt))
    return cmap, cells, mus, ok


class StagJaxTwin:
    """The frozen template of a :class:`~lumenairy.elements.pmm.PMM2DStackPure`
    (shared grid, source set) and its differentiable solve.

    Built from a CONCRETE stack: every discrete decision the NumPy solve
    takes from values is taken here, once (module docstring).  :meth:`solve`
    then evaluates the cascade on a parameter dictionary (a pytree of JAX or
    NumPy values; :meth:`params` returns the reference one) and is pure, so
    ``jax.jit`` / ``jax.grad`` compose with it.

    ``geometry='auto'`` makes the template MAPPED (quadrature assembly, the
    route a traced map needs) whenever the stack has a map or shape layers,
    and UNMAPPED otherwise; ``'mapped'`` forces a stack of rectangles (or an
    eps_cell stack) onto an identity transfinite map on its own walls so that
    its wall positions can be traced; ``'static'`` keeps the stack's own route
    and refuses a traced map."""

    def __init__(self, stack, *, geometry="auto"):
        from ._curvemap import TransfiniteMap
        fn = "PMM2DStackPure(backend='jax')"
        if stack._src is None:
            raise ValueError(f"{fn}: call set_source(...) first.")
        if not stack._layers:
            raise ValueError(f"{fn}: add at least one layer.")
        if stack.layer_grids != "shared":
            raise NotImplementedError(
                f"{fn}: layer_grids='per-layer' has no JAX twin -- the L2 "
                f"mortar between per-layer grids (and per-layer MAPS, Phase "
                f"E2 of the curved-cell plan) is NumPy-only.  Use the default "
                f"layer_grids='shared'.")
        for k, L in enumerate(stack._layers):
            if not _slant_is_zero(L.get("slant")):
                raise NotImplementedError(
                    f"{fn}: layer {k + 1} is SLANTED; slant runs the "
                    f"out-of-plane first-order generator, which has no JAX "
                    f"twin (slant under a map is Phase E1 of the curved-cell "
                    f"plan).  Use NumPy inputs.")
            for key, uni in (("eps33", True), ("eps", L.get("eps_uniform",
                                                             True)),
                             ("eps_cell", False),
                             ("mu", L.get("mu_uniform", True))):
                c = L.get(key)
                if c is None or np.ndim(c) == 0:
                    continue
                a = np.asarray(c)
                t33 = (a[None, None] if (uni and a.shape == (3, 3)) else
                       a if (not uni and a.ndim == 4) else None)
                if t33 is not None and _tile_is_offplane(t33):
                    raise NotImplementedError(
                        f"{fn}: layer {k + 1} carries an OUT-OF-PLANE tensor "
                        f"(e_xz / e_yz / e_zx / e_zy); the out-of-plane "
                        f"first-order generator has no JAX twin (out-of-plane "
                        f"tensors under a map are Phase E1 of the curved-cell "
                        f"plan).  Use NumPy inputs.")
        if geometry not in ("auto", "mapped", "static"):
            raise ValueError(f"{fn}: geometry must be 'auto', 'mapped' or "
                             f"'static', got {geometry!r}.")
        self.stack = stack
        px, py = stack.period_x, stack.period_y
        self.px, self.py = px, py
        Nx, Ny = stack._grid if stack._grid is not None else (2, 2)
        self.Nx, self.Ny = Nx, Ny
        M = stack.M
        self.M = M
        self.n_orders = stack.n_orders
        (wl, k0, kx0, ky0, a0x, a0y,
         eps_sup, eps_sub) = stack._source_prep()
        self.wl, self.k0, self.kx0, self.ky0 = wl, k0, kx0, ky0
        self.a0x, self.a0y = a0x, a0y
        self.eps_sup0, self.eps_sub0 = eps_sup, eps_sub
        cmap = stack.cmap
        # THE SHAPE LAYERS: the reference merge, with its intermediates (the
        # structure a traced shape parameter is replayed on)
        self.shape_parts: dict[str, Any] | None = None
        self.shape_idx = []
        if stack._shapes_map:
            from .shapes2d import _merge
            self.shape_idx = [k for k, L in enumerate(stack._layers)
                              if "shapes" in L]
            self.shape_ref_layers = [
                (stack._layers[k]["shapes"],
                 stack._layers[k]["background_eps"],
                 stack._layers[k].get("background_mu"))
                for k in self.shape_idx]
            (_U, _V, cm_s, _c, ident, _m, parts) = _merge(
                px, py, [(f"layer {k + 1}",) + lay for k, lay in
                         zip(self.shape_idx, self.shape_ref_layers)],
                parts=True)
            self.shape_parts = parts
            if geometry != "static":
                # the merged map itself (an identity transfinite map for a
                # rectangles-only stack: the route a traced wall needs)
                cmap = cm_s
        if cmap is None and stack._shape_walls is not None:
            gx, gy = stack._shape_walls
        elif cmap is None:
            gx, gy = Nx, Ny
        else:
            gx, gy = cmap.u_walls, cmap.v_walls
        if cmap is None and geometry == "mapped":
            # a rectangles-only (or eps_cell) stack whose wall positions are
            # to be traced: the identity TRANSFINITE map on its own walls --
            # the quadrature route, whose vertex images a trace can move
            bxw = TS.Basis1D(px, gx, M).xb
            byw = TS.Basis1D(py, gy, M).xb
            cmap = TransfiniteMap(bxw, byw)
            gx, gy = cmap.u_walls, cmap.v_walls
        self.mapped = cmap is not None
        self.cmap_ref = cmap
        self.gx, self.gy = gx, gy
        mkw = {} if cmap is None else {"cmap": cmap}
        self._mkw = mkw
        # the homogeneous reference solver: the shared geometric eig
        self.sol_h = TS.Granet2DTransverseE(px, py, gx, gy, M,
                                            np.full((Nx, Ny), eps_sup),
                                            alpha0x=a0x, alpha0y=a0y, k0=k0,
                                            **mkw)
        self.geom_ref = TS._homog_geom_cache(self.sol_h)
        bx, by = self.sol_h.bx, self.sol_h.by
        self.bx, self.by = bx, by
        self.qq = (Nx * (M - 1)) * (Ny * (M - 1))
        # per-layer reference solvers (deduplicated like the stack's eig cache)
        self.layers = []
        refs: dict[Any, Any] = {}
        for L in stack._layers:
            rec: dict[str, Any] = dict(kind=L["kind"],
                                       thickness=L["thickness"])
            if L["kind"] == "uniform":
                rec["eps"] = L["eps"]
            else:
                mcell = None
                if L["kind"] == "uniform_tensor":
                    cell = np.ascontiguousarray(
                        np.broadcast_to(L["eps33"], (Nx, Ny, 3, 3)))
                    rec["eps"] = L["eps33"]
                elif L["kind"] == "magnetic":
                    from .stack2d_pure import _as_layer_cell
                    cell = _as_layer_cell(L["eps"], L["eps_uniform"], Nx, Ny)
                    mcell = _as_layer_cell(L["mu"], L["mu_uniform"], Nx, Ny)
                    rec.update(eps=L["eps"], eps_uniform=L["eps_uniform"],
                               mu=L["mu"], mu_uniform=L["mu_uniform"])
                else:
                    cell = L["eps_cell"]
                    rec["eps"] = cell
                rkey = ((cell.shape, cell.tobytes()) if mcell is None else
                        (cell.shape, cell.tobytes(), mcell.shape,
                         mcell.tobytes()))
                ref = refs.get(rkey)
                if ref is None:
                    ref = refs[rkey] = TS.Granet2DTransverseE(
                        px, py, gx, gy, M, cell, alpha0x=a0x, alpha0y=a0y,
                        k0=k0, mu_cell=mcell, **mkw)
                rec["ref"] = ref
                rec["ref_key"] = rkey
            self.layers.append(rec)
        # the order set and the far field
        ox = np.arange(-self.n_orders, self.n_orders + 1)
        self.ox = self.oy = ox
        self.order_x = np.tile(ox, len(ox))
        self.order_y = np.repeat(ox, len(ox))
        self.Nfo = self.order_x.size
        self.p0 = int(np.where((self.order_x == 0) & (self.order_y == 0))[0][0])
        self.kxv = kx0 + self.order_x * (wl / px)
        self.kyv = ky0 + self.order_y * (wl / py)
        if not self.mapped:
            self.P_far = TS._far_projector_2d(bx, by, ox, ox, a0x, a0y)
            self.far_nq: dict[Any, int] | None = None
            self.inc_nq: dict[Any, int] | None = None
        else:
            self.far_nq, self.inc_nq = {}, {}
            self.P_far = TS._far_projector_mapped(bx, by, ox, ox, a0x, a0y,
                                                  cmap, record=self.far_nq)
            TS._stag_incident_load_mapped(bx, by, cmap, a0x, a0y, (1.0, 0.0),
                                          record=self.inc_nq)
        self._jit_cache = {}

    # ------------------------------------------------------------- params
    def params(self):
        """The REFERENCE parameter dictionary: ``n_superstrate``,
        ``n_substrate``, and per layer ``thickness``, ``eps`` (a scalar or
        ``(3, 3)`` for a uniform layer, the ``(Nx, Ny)`` / ``(Nx, Ny, 3, 3)``
        cell of a patterned one) and ``mu`` (``None``, or as ``eps``).  Every
        leaf may be replaced by a JAX value of the same shape."""
        st = self.stack
        layers = []
        for k, rec in enumerate(self.layers):
            d = dict(thickness=rec["thickness"], eps=rec["eps"],
                     mu=rec.get("mu"))
            if k in self.shape_idx:
                shp, bg, bgm = self.shape_ref_layers[self.shape_idx.index(k)]
                d = dict(thickness=rec["thickness"], shapes=shp,
                         background_eps=bg, background_mu=bgm)
            layers.append(d)
        return dict(n_superstrate=st.n_sup, n_substrate=st.n_sub,
                    layers=layers)

    # -------------------------------------------------------------- solve
    def solve(self, params=None, *, cmap=None):
        """The differentiable cascade: ``(orders, R(2, Nfo), T(2, Nfo),
        jones(2, 2))`` with JAX ``R`` / ``T`` / ``jones``, the
        :meth:`PMM2DStackPure.solve` contract.

        ``params`` (default :meth:`params`) carries the materials, the
        thicknesses and the half-space indices; ``cmap`` a TRACED coordinate
        map on this template's frozen ``(u, v)`` grid
        (:meth:`~lumenairy.elements.pmm._curvemap.TransfiniteMap._traced`),
        or ``None`` for the reference map.  A traced map whose image folds at
        a node returns NaN (gate E3-4)."""
        import jax.numpy as jnp

        from ..rcwa import _require_jax_x64
        from ..rcwa._core import (
            _interface_smatrix,
            _project_efficiency,
            _propagation_smatrix,
            _redheffer_star,
        )
        _require_jax_x64("PMM2DStackPure.solve (JAX)")
        xp = jnp
        cj = jnp.complex128
        p = self.params() if params is None else params
        ok_topo = None
        p, cmap_s, ok_topo = self._shape_params(p, cmap)
        if cmap_s is not None:
            cmap = cmap_s
        traced_map = cmap is not None
        if traced_map and not self.mapped:
            raise ValueError(
                "PMM2DStackPure (JAX): this template is UNMAPPED, so a traced "
                "map has no quadrature route to enter -- build the twin with "
                "geometry='mapped'.")
        cm = self.cmap_ref if cmap is None else cmap
        if self.kx0 != 0.0 or self.ky0 != 0.0:
            from ._curvemap import _any_traced
            ns = p["n_superstrate"]
            if _any_traced(ns) or complex(ns) != complex(self.stack.n_sup):
                raise NotImplementedError(
                    "PMM2DStackPure (JAX): at OBLIQUE incidence the "
                    "superstrate index sets the in-plane wavevector -- the "
                    "Bloch glue of the frozen basis -- so it cannot be traced "
                    "or changed here (trace it at normal incidence, or "
                    "rebuild the twin).")
        n_sup = jnp.asarray(p["n_superstrate"]).astype(cj)
        n_sub = jnp.asarray(p["n_substrate"]).astype(cj)
        eps_sup = n_sup ** 2
        eps_sub = n_sub ** 2
        k0 = self.k0
        qq = self.qq
        Nx, Ny = self.Nx, self.Ny

        # ---- STAGE 1: every eig PROBLEM of the solve (the shared geometric
        # pencil when the map is traced; one per distinct patterned layer)
        # and every eigenpair-free quantity.  The eigs and their CONSUMER
        # (stage 2, below) run through ``rcwa._jax_eig_cluster_adjoint``,
        # whose reverse rule is correct at DEGENERATE eigenvalue clusters
        # (the four-fold symmetric cell, verifier V-E3-1); the forward values
        # are those of the plain composition.
        from ..rcwa._core import _jax_eig_cluster_adjoint
        problems: list[Any] = []
        anchors: list[Any] = []
        sh = None
        if traced_map:
            sh = _shadow(self.sol_h, xp,
                         jnp.full((Nx, Ny), self.eps_sup0, dtype=cj), None,
                         cm)
            problems.append((sh.Stt - sh.Schur, -sh.Rmat))
            # the consumer's branch points in g2_geo: g2 = g2_geo + eps = 0
            # for every homogeneous region (reference values: the anchors
            # only size the degenerate-cluster lift)
            anchors.append(tuple(-complex(e) for e in self._homog_eps_ref()))
        lay_ix: list[Any] = []     # per layer: None (uniform) or the problem
        seen: dict[Any, Any] = {}
        for rec, lp in zip(self.layers, p["layers"]):
            if rec["kind"] == "uniform":
                lay_ix.append(None)
                continue
            # dedupe on the reference key AND the identity of the traced
            # leaves (two layers sharing one traced cell share one eig)
            dkey = (rec["ref_key"], id(lp["eps"]), id(lp.get("mu")))
            hit = seen.get(dkey)
            if hit is None:
                eps_c, mu_c = self._layer_cells(rec, lp)
                shl = _shadow(rec["ref"], xp, eps_c, mu_c, cm)
                problems.append((shl.Lmat, -shl.Rmat))
                anchors.append((0.0,))            # q = sqrt(g2)
                hit = seen[dkey] = (len(problems) - 1, shl)
            lay_ix.append(hit)
        if not self.mapped:
            P1, P2 = self.P_far
            P12 = P21 = None
        elif traced_map:
            P1, P2, P12, P21 = TS._far_projector_mapped(
                self.bx, self.by, self.ox, self.oy, self.a0x, self.a0y,
                cm, xp=xp, nq_cells=self.far_nq)
        else:
            P1, P2, P12, P21 = self.P_far
        Nfo, p0 = self.Nfo, self.p0
        kxv, kyv = self.kxv, self.kyv
        kz_ref, kz_trn, kz_inc, safe_r, safe_t = TS._pmm2d_order_kz(
            eps_sup, eps_sub, kxv, kyv, self.kx0, self.ky0, xp=xp)
        poison = None
        if traced_map or ok_topo is not None:
            ok = _min_detj(self.sol_h, cm, xp) > 0.0
            if ok_topo is not None:
                ok = ok & ok_topo
            # a MULTIPLICATIVE poison, so the GRADIENT is NaN too (a
            # jnp.where would hand the cotangent to the finite branch and
            # return a silent zero gradient at the event)
            poison = xp.where(ok, 1.0, xp.nan)

        def consumer(eigs):
            """STAGE 2: everything that depends on an eigenpair."""
            # ---- the shared geometric eig (half-spaces + uniform layers)
            if traced_map:
                g2_geo, W0 = eigs[0]
                geom = TS._homog_geom_from_eig(sh, g2_geo, W0, xp=xp)
            else:
                geom = self.geom_ref
            Wsup, Vsup, _l = TS._homog_region_modes(geom, eps_sup, xp=xp)
            Wsub, Vsub, _l = TS._homog_region_modes(geom, eps_sub, xp=xp)
            # (a template without a traced map keeps W0 a NumPy constant; the
            # cascade's array-namespace dispatch needs ONE backend)
            Wsup, Vsup, Wsub, Vsub = (jnp.asarray(a) for a in (Wsup, Vsup,
                                                               Wsub, Vsub))

            # ---- per-layer modes
            modes = []
            cache: dict[Any, Any] = {}
            for lp, hit in zip(p["layers"], lay_ix):
                t = lp["thickness"]
                if hit is None:
                    W, V, lam = TS._homog_region_modes(
                        geom, jnp.asarray(lp["eps"]).astype(cj), xp=xp)
                else:
                    k, shl = hit
                    got = cache.get(k)
                    if got is None:
                        g2, Wl = eigs[k]
                        W, V, lam, _g = TS._region_modes_from_eig(
                            shl, g2, Wl, xp=xp)
                        got = cache[k] = (W, V, lam)
                    W, V, lam = got
                modes.append((jnp.asarray(W), jnp.asarray(V),
                              jnp.asarray(lam), t))

            # ---- the square Redheffer cascade (rcwa._core's algebra, as the
            # NumPy stack's)
            nlay = len(modes)
            ifc = [_interface_smatrix(Wsup, Vsup, modes[0][0], modes[0][1])]
            for i in range(1, nlay):
                ifc.append(_interface_smatrix(modes[i - 1][0],
                                              modes[i - 1][1],
                                              modes[i][0], modes[i][1]))
            ifc.append(_interface_smatrix(modes[-1][0], modes[-1][1], Wsub,
                                          Vsub))
            S = ifc[0]
            for i in range(nlay):
                S = _redheffer_star(S, _propagation_smatrix(
                    modes[i][2], k0 * modes[i][3]))
                S = _redheffer_star(S, ifc[i + 1])
            S11, _S12, S21, _S22 = S

            # ---- far field + incident decomposition
            if not self.mapped:
                Hsup = TS._pmm2d_project_orders(P1, P2, Wsup, qq, xp=xp)
                Hsub = TS._pmm2d_project_orders(P1, P2, Wsub, qq, xp=xp)
            else:
                Hsup = TS._pmm2d_project_orders(P1, P2, Wsup, qq, P12, P21,
                                                xp=xp)
                Hsub = TS._pmm2d_project_orders(P1, P2, Wsub, qq, P12, P21,
                                                xp=xp)
            if self.mapped:
                cinc_map = TS._stag_incident_coeffs_mapped(
                    geom, self.bx, self.by, cm, self.a0x, self.a0y,
                    H0=Hsup[np.array([p0, Nfo + p0]), :], xp=xp,
                    nq_cells=self.inc_nq)
            else:
                cinc_map = self._unmapped_cinc()
            R_rows, T_rows, j_cols = [], [], []
            for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
                long_inc = self.kx0 * ex0 + self.ky0 * ey0
                einc_sq = 1.0 + (long_inc / kz_inc) ** 2
                cinc = cinc_map[:, col]
                r_ord = Hsup @ (S11 @ cinc)
                t_ord = Hsub @ (S21 @ cinc)
                rx, ry = r_ord[:Nfo], r_ord[Nfo:]
                tx, ty = t_ord[:Nfo], t_ord[Nfo:]
                rz = -(kxv * rx + kyv * ry) / safe_r
                tz = -(kxv * tx + kyv * ty) / safe_t
                Re, Te = _project_efficiency(xp, kz_ref, kz_trn, kz_inc,
                                             rx, ry, rz, tx, ty, tz, einc_sq)
                R_rows.append(Re)
                T_rows.append(Te)
                j_cols.append(xp.stack([rx[p0], ry[p0]]))
            R_eff = xp.stack(R_rows)
            T_eff = xp.stack(T_rows)
            jmat = xp.stack(j_cols, axis=1)
            if poison is not None:
                R_eff = R_eff * poison
                T_eff = T_eff * poison
                jmat = jmat * poison
            return R_eff, T_eff, jmat

        R_eff, T_eff, jmat = _jax_eig_cluster_adjoint(
            _stag_geneig_jax, problems, consumer,
            gap_rel=_E3_EIG_CLUSTER_GAP_REL,
            split_rel=_E3_EIG_CLUSTER_SPLIT_REL, anchors=anchors)
        orders2d = np.stack([self.order_x, self.order_y], axis=1)
        return orders2d, R_eff, T_eff, jmat

    # ------------------------------------------------------------ helpers
    def _homog_eps_ref(self):
        """The reference scalar permittivities of every homogeneous region
        that is solved from the shared geometric eig (the half-spaces and the
        uniform scalar layers)."""
        out = [self.eps_sup0, self.eps_sub0]
        out += [rec["eps"] for rec in self.layers if rec["kind"] == "uniform"
                and np.ndim(rec["eps"]) == 0]
        return out

    def _shape_params(self, p, cmap):
        """Resolve the SHAPE layers of ``p``: when any shape (or a
        background) differs from the reference objects, replay the merge on
        the traced values (:func:`_traced_shape_merge`) and return ``(p'
        with every shape layer's eps / mu cell filled in, the traced map or
        None, the in-trace topology flag)``.  When every shape parameter is
        CONCRETE the reference topology is also re-derived by the NumPy merge
        and a change RAISES (gate E3-4, the refusal arm)."""
        if self.shape_parts is None:
            return p, None, None
        t_layers, geo_changed, mat_changed = [], False, False
        for k, ref in zip(self.shape_idx, self.shape_ref_layers):
            lp = p["layers"][k]
            shp = tuple(lp.get("shapes", ref[0]))
            bg = lp.get("background_eps", ref[1])
            bgm = lp.get("background_mu", ref[2])
            if len(shp) != len(ref[0]) or any(
                    a is not b for a, b in zip(shp, ref[0])):
                geo_changed = True
            if bg is not ref[1] or bgm is not ref[2]:
                mat_changed = True
            t_layers.append((shp, bg, bgm))
        if not (geo_changed or mat_changed):
            # the reference shapes: the reference cells
            layers = list(p["layers"])
            for k in self.shape_idx:
                rec = self.layers[k]
                layers[k] = dict(layers[k], eps=rec["eps"], mu=rec.get("mu"))
            return dict(p, layers=layers), None, None
        if geo_changed and cmap is not None:
            raise ValueError(
                "PMM2DStackPure (JAX): pass either traced shapes or an "
                "explicit traced cmap, not both.")
        if geo_changed:
            self._check_topology_concrete(t_layers)
        cm_t, cells, mus, ok = _traced_shape_merge(
            self.shape_parts, self.shape_ref_layers, t_layers, self.px,
            self.py, self.cmap_ref)
        layers = list(p["layers"])
        for k, cell, mcell in zip(self.shape_idx, cells, mus):
            d = dict(layers[k])
            d["eps"] = cell
            d["mu"] = mcell
            layers[k] = d
        p = dict(p, layers=layers)
        return p, (cm_t if geo_changed else None), (ok if geo_changed
                                                     else None)

    def _check_topology_concrete(self, t_layers):
        """The REFUSAL arm of gate E3-4: when every geometric parameter of
        the traced shapes is concrete (an eager call), re-run the NumPy merge
        at those values and refuse a topology that differs from the frozen
        one (wall count, wall owners, singular vertices, grid squaring).
        Under ``jax.jit`` / ``jax.grad`` the values are abstract and the
        in-trace NaN guard stands in."""
        from ._curvemap import _any_traced
        for shp, _b, _m in t_layers:
            for sh in shp:
                if _any_traced(*[getattr(sh, k) for k in sh._GEOM]):
                    return
        from .shapes2d import _merge

        def geometry_only(sh):
            # the merge paints materials too: a TRACED eps / mu (with
            # concrete geometry) is replaced by a concrete stand-in of the
            # same kind -- the topology does not depend on it
            if not _any_traced(sh.eps, sh.mu):
                return sh
            c = copy.copy(sh)
            object.__setattr__(c, "eps", 1.0 + 0j if np.ndim(sh.eps) == 0
                               else np.eye(3, dtype=complex))
            if sh.mu is not None:
                object.__setattr__(c, "mu", 1.0 + 0j if np.ndim(sh.mu) == 0
                                   else np.eye(3, dtype=complex))
            return c
        conc = [(tuple(geometry_only(sh) for sh in shp),
                 1.0 if _any_traced(b) else b,
                 None if m is None else (1.0 if _any_traced(m) else m))
                for shp, b, m in t_layers]
        try:
            U, V, cm, _c, _i, _m, parts = _merge(
                self.px, self.py, [(f"layer {k + 1}",) + lay for k, lay in
                                   zip(self.shape_idx, conc)],
                parts=True)
        except ValueError as exc:
            raise ValueError(
                f"PMM2DStackPure (JAX): the shapes at these parameter values "
                f"are refused by the merge ({exc}) -- outside the twin's "
                f"frozen topology.") from None
        ref = self.shape_parts
        assert ref is not None              # a shapes template (caller)

        def owners(pt, g):
            # wall owners as ITEM INDICES (the owner labels carry the shape
            # parameters, which legitimately change)
            idx = {who: k for k, (who, _s, _l) in enumerate(pt["items"])}
            return [sorted(idx.get(w, -1) for w in o) for o in pt[g].owners]
        same = (U.size == self.cmap_ref.u_bounds.size
                and V.size == self.cmap_ref.v_bounds.size
                and owners(parts, "gu") == owners(ref, "gu")
                and owners(parts, "gv") == owners(ref, "gv")
                and parts["tm"].singular_vertices
                == ref["tm"].singular_vertices)
        if not same:
            raise ValueError(
                "PMM2DStackPure (JAX): these shape parameters change the "
                "TOPOLOGY of the merged wall grid (walls merging or "
                "separating, a different number of segments, singular "
                "vertices appearing or vanishing) -- a non-differentiable "
                "event of the discretisation.  The twin is frozen at its "
                "reference topology: rebuild it (backend='jax') at the new "
                "geometry.")

    def _layer_cells(self, rec, lp):
        """The (possibly traced) ``(eps_cell, mu_cell)`` of a non-uniform-
        scalar layer on the template grid."""
        import jax.numpy as jnp
        cj = jnp.complex128
        Nx, Ny = self.Nx, self.Ny
        if rec["kind"] == "uniform_tensor":
            e = jnp.asarray(lp["eps"]).astype(cj)
            return jnp.broadcast_to(e, (Nx, Ny, 3, 3)), None
        if rec["kind"] == "magnetic":
            def cellof(v, uniform):
                a = jnp.asarray(v).astype(cj)
                if not uniform:
                    return a
                if a.ndim == 0:
                    return jnp.full((Nx, Ny), a, dtype=cj)
                return jnp.broadcast_to(a, (Nx, Ny, 3, 3))
            return (cellof(lp["eps"], rec["eps_uniform"]),
                    cellof(lp["mu"], rec["mu_uniform"]))
        return jnp.asarray(lp["eps"]).astype(cj), None

    def _unmapped_cinc(self):
        """The unmapped incident modal amplitudes: the shipped least-squares
        overlap on the reference half-space modes -- geometry-only and
        eps-free (``W0`` is the eps-free geometric eig), so a NumPy CONSTANT
        of the template, bit for bit the NumPy stack's."""
        hit = getattr(self, "_cinc_unmapped", None)
        if hit is not None:
            return hit
        P1, P2 = self.P_far
        W0 = self.geom_ref[0]
        Hsup = TS._pmm2d_project_orders(P1, P2, W0, self.qq)
        delta = ((self.order_x == 0) & (self.order_y == 0)).astype(_C)
        cols = []
        for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
            rhs = np.concatenate([ex0 * delta, ey0 * delta])
            cols.append(_guarded_lstsq(
                Hsup, rhs, "PMM2DStackPure far-field Rayleigh projection"))
        self._cinc_unmapped = np.stack(cols, axis=1)
        return self._cinc_unmapped


def _pmm_jones_2d_staggered_jax(period_x, period_y, eps_cell, n_substrate,
                                n_superstrate, depth, wavelength, *, mu_cell,
                                M, n_orders, theta, phi, slant, cmap, shapes,
                                background_eps, background_mu,
                                reference_shapes, mu_superstrate,
                                mu_substrate, max_pencil_dof):
    """``pmm_jones_2d_staggered(..., backend='jax')``: the one-layer stack
    built on CONCRETE stand-ins for every traced argument, then solved by
    its twin with the traced values as parameters."""
    from ._curvemap import _any_traced
    from .stack2d_pure import PMM2DStackPure
    fn = "pmm_jones_2d_staggered(backend='jax')"
    TS._require_nonmagnetic_halfspace(fn, mu_superstrate, mu_substrate)
    if _any_traced(period_x, period_y, wavelength, theta, phi):
        raise NotImplementedError(
            f"{fn}: period_x / period_y / wavelength / theta / phi must be "
            f"CONCRETE -- they set the Bloch glue and the wall grid of the "
            f"basis, which the twin freezes.  Trace materials, the depth, the "
            f"half-space indices or shape parameters instead.")
    if not _slant_is_zero(_norm_slant_pair(slant, fn)):
        raise NotImplementedError(
            f"{fn}: slant= has no JAX twin (slant under a map is Phase E1 of "
            f"the curved-cell plan); use backend='numpy'.")
    oblique = float(theta) != 0.0

    def conc(v, stand_in):
        return stand_in if _any_traced(v) else v
    if oblique and _any_traced(n_superstrate):
        raise NotImplementedError(
            f"{fn}: a traced n_superstrate at OBLIQUE incidence is not "
            f"differentiable here -- it sets the in-plane wavevector, i.e. "
            f"the Bloch glue of the basis, which the twin freezes.  Trace it "
            f"at normal incidence only.")
    n_sub_c = conc(n_substrate, 1.5)
    n_sup_c = conc(n_superstrate, 1.0)
    depth_c = conc(depth, 1.0)
    kw = dict(n_superstrate=n_sup_c, n_substrate=n_sub_c, n_modes=M,
              n_orders=n_orders, backend="jax")
    if shapes is not None or background_eps is not None \
            or background_mu is not None:
        shapes = list(shapes)
        traced_shapes = any(
            _any_traced(*[getattr(sh, k) for k in sh._GEOM], sh.eps,
                        sh.mu) for sh in shapes)
        if traced_shapes and reference_shapes is None:
            raise ValueError(
                f"{fn}: the shapes carry JAX values; pass reference_shapes= "
                f"(the same shapes at CONCRETE values -- the reference the "
                f"wall grid, its topology and the quadrature are frozen at).")
        ref_shapes = list(reference_shapes) if traced_shapes else shapes
        bg_c = conc(background_eps, 1.0 if np.ndim(background_eps) == 0
                    else np.eye(3))
        bgm_c = None if background_mu is None else conc(
            background_mu, 1.0 if np.ndim(background_mu) == 0 else np.eye(3))
        st = PMM2DStackPure(period_x, period_y, **kw)
        st.add_layer(float(depth_c), shapes=ref_shapes, background_eps=bg_c,
                     background_mu=bgm_c, max_pencil_dof=max_pencil_dof)
        st.set_source(float(wavelength), theta=float(theta), phi=float(phi))
        p = st.jax_params()
        lp = p["layers"][0]
        lp.update(thickness=depth, shapes=shapes, background_eps=background_eps,
                  background_mu=background_mu)
    else:
        cell = eps_cell
        if _any_traced(eps_cell):
            cell = np.full(np.shape(eps_cell), 2.0 + 0.0j)
            if cell.ndim == 4:
                cell = np.broadcast_to(2.0 * np.eye(3), cell.shape).copy()
        mcell = mu_cell
        if mu_cell is not None and _any_traced(mu_cell):
            mcell = np.ones(np.shape(mu_cell), dtype=complex)
            if mcell.ndim == 4:
                mcell = np.broadcast_to(np.eye(3, dtype=complex),
                                        mcell.shape).copy()
        st = PMM2DStackPure(period_x, period_y,
                            **(kw if cmap is None else dict(kw, cmap=cmap)))
        if mcell is None:
            st.add_layer(float(depth_c), eps_cell=cell,
                         max_pencil_dof=max_pencil_dof)
        else:
            st.add_layer(float(depth_c), eps_cell=cell, mu_cell=mcell,
                         max_pencil_dof=max_pencil_dof)
        st.set_source(float(wavelength), theta=float(theta), phi=float(phi))
        p = st.jax_params()
        lp = p["layers"][0]
        lp["thickness"] = depth
        if st._layers[0]["kind"] == "magnetic":
            lp["eps"], lp["mu"] = eps_cell, mu_cell
        else:
            lp["eps"] = eps_cell
    p["n_substrate"] = n_substrate
    p["n_superstrate"] = n_superstrate
    return st.solve(params=p)
