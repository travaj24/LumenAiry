"""Shape primitives for curved cells in the pure staggered 2-D PMM.
=====================================================================

What this module is for
-----------------------
The pure staggered 2-D PMM (:func:`~lumenairy.elements.pmm.pmm_jones_2d_staggered`,
:class:`~lumenairy.elements.pmm.PMM2DStackPure`) divides the unit cell into a
grid of rectangles (the **wall grid**) and expands the field in each rectangle
in polynomials.  On its own every material boundary must be one of the grid's
straight walls, so a circular pillar is a staircase of rectangles.  A
**coordinate map** bends that grid so that a circle, an ellipse, a rounded
corner or a sinusoidal wall becomes an EXACT grid line
(:mod:`lumenairy.elements.pmm._curvemap`).  Laying out such a map by hand
means choosing the walls, the curved edges and the moved vertices yourself --
and getting them wrong costs either a different device or several decades of
convergence (planning finding F5, Phase A verifier D6).  The SHAPE PRIMITIVES
here take PHYSICAL geometry (centres, sizes, radii, in metres) and do that
layout for you:

* :class:`Rect` -- an axis-aligned rectangle (a sharp-cornered pillar, a
  stripe when it spans the period);
* :class:`FilletRect` -- a rectangle with its four corners rounded to a
  radius ``r``;
* :class:`Circle` -- a circular disk;
* :class:`Ellipse` -- an elliptical disk (axis-aligned, or rotated by up to
  45 degrees);
* :class:`SinusoidalWall` -- an interface that wiggles sinusoidally across
  the cell (or, with ``width=``, a sinusoidal ridge of constant width).

Words used below
----------------
* **Wall grid / ``(u, v)``** -- the solver's straight rectangular grid, in the
  solver's own coordinates ``(u, v)``; the map sends it to the physical
  ``(x, y)`` plane.  Where nothing is curved the map is the identity and
  ``(u, v) = (x, y)``.
* **Preimage** -- the ``(u, v)`` position whose image is a given physical
  point.  Every primitive places its walls so that each physical boundary IS
  the image of a grid line, with the corners of every shape at their own
  physical positions (identity-mapped vertices), which is what keeps the
  solver's convergence fast (F5) and the device the one you described (D6).
* **Hard edge** -- a piece of a shape's physical outline, carried by one grid
  line: a straight side, an arc, a piece of an ellipse or of a sinusoid.
  Every hard edge is represented EXACTLY by the map.
* **Tangency point** -- where a fillet's arc meets a straight side.  The
  curvature jumps there, so it must be a grid vertex (a FilletRect therefore
  needs a 5 x 5 wall grid; planning section 3.4).
* **Singular vertex** -- a cell corner where the map pinches (``det J = 0``):
  the four 45-degree points of a circle, an ellipse or a fillet.  They are
  unavoidable on a tensor grid, measured harmless, and integrated by the
  solver's corner (Duffy) rule.

How shapes combine (painting, merging, refusing)
------------------------------------------------
Within ONE layer the shapes are PAINTED in order onto ``background_eps``: a
later shape covers an earlier one, so a circular hole in a slab is
``[Rect(..., eps=4.0), Circle(..., eps=1.0)]``.  Across the LAYERS of a stack
the map is shared (it is owned by the stack): :func:`compile_shapes` (one
layer) and :meth:`~lumenairy.elements.pmm.PMM2DStackPure.add_layer`
(``shapes=``, every layer) take the UNION of every shape's walls as the wall
grid and give every hard edge its exact curve.  Two layers may share a curve
(the same circle in two layers is one curve).  The merge REFUSES, naming both
shapes and their layers:

* two outlines that CROSS in plan view (a circle in one layer cut by the edge
  of a rectangle in another) -- one map cannot make both exact; with
  ``layer_grids='per-layer'`` shapes in DIFFERENT layers keep their own maps
  instead (Phase E2, the curved mortar);
* two different curves on one cell edge, or two different images of one grid
  vertex (a filleted corner in one layer over a sharp corner in another --
  "different curves in one cell");
* a merged map that FOLDS (``det J <= 0``) in a cell between two outlines
  that come too close -- including shapes that are far apart: a wall of one
  shape that crosses the arc "bulge" of another (between its 45-degree
  points and its extreme) is kept straight and folds the cell.  Measured
  (Phase C verifier V-D3): two circles side by side in one row with radius
  ratio between 1 and 1.414, or equal circles whose centres differ by
  0.02-0.05 of the period in y, raise.  With
  ``PMM2DStackPure(..., layer_grids='per-layer')`` shapes in DIFFERENT
  layers then keep their own maps (Phase E2, the curved mortar); within one
  layer they still raise;
* a segment narrower than the staggered solver's SLIVER CONTRACT (1e-3 of the
  period, ``twod_staggered._STAG_MIN_SEG_FRAC``), naming the shapes whose
  walls bound it.

Two outlines that do NOT cross may share the cells between them -- a cell
whose left edge is one circle's arc and whose right edge is its neighbour's is
an ordinary transfinite cell -- and a shape may sit inside another (a small
square over a large disk in the next layer).  The basis needs a SQUARE grid
(``Nx == Ny``); the shorter axis is squared up by halving its widest segment
(:class:`~lumenairy.elements.pmm._curvemap.RefinedMap`: the added wall
subdivides cells without moving any outline).

What a fillet buys (geometry fidelity, not speed)
-------------------------------------------------
Rounding a pillar's corners changes the device -- at radius 0.2 of the side
the zeroth-order transmission moves by 7.4e-3, far above the solver's
convergence level -- and a same-area square is no substitute (it moves the
reflection the wrong way).  It does NOT make the diffraction efficiencies
converge faster: they are capped near 1e-5 per rung by the pillar's top and
bottom RIM, which no in-plane rounding removes (planning section 3.4, Phase B
gates B6 / B7).  Use a fillet because the fabricated device has one.

Materials: anisotropic and magnetic shapes
------------------------------------------
Every primitive takes ``eps`` as a scalar or a ``(3, 3)`` BLOCK-FORM tensor
(in-plane anisotropy: a liquid-crystal director lying in the plane, a
gyrotropic ``e12 = -e21``; ``e13 = e23 = e31 = e32 = 0``) and an optional
relative permeability ``mu=`` of the same two kinds.  The layer's
``background_mu=`` (default 1) fills the rest; a shape without ``mu`` paints
``mu = 1``.  Under a curved map a tensor is carried through the map by the
congruence ``sqrt(g) J^-1 eps J^-T`` (and ``chi_t = J^T mu_t^-1 J /
sqrt(g)``) evaluated at every quadrature node -- the effective tensor needs
the Jacobian itself, not only the metric a scalar uses (Phase D,
``docs/audits/BUILD_PMM2D_CURVED_D_2026_10_03.md``).  A liquid-crystal-filled
circular hole is ``[Rect(..., eps=4.0), Circle(..., eps=lc_tensor)]``.

Known limits
------------
* OUT-OF-PLANE tensors (a tilted director, ``e13 != 0``) under a curved map
  are Phase E and raise; ``slant=`` and a STACK-wide ``cmap=`` with
  ``layer_grids='per-layer'`` raise (per-layer maps are Phase E2); with
  rectangles only (no map) an out-of-plane tensor is accepted.
* A FilletRect's flat sides lie on the grid lines through its 45-degree
  points (the Phase B layout the gates were measured on), so another shape's
  straight edge cannot coincide with a fillet's flat side (it would close a
  zero-width cell and raises as a fold).
* Shapes must lie inside the unit cell (no wrapping across the periodic
  seam).

Plan: ``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.3;
build: ``docs/audits/BUILD_PMM2D_CURVED_C_2026_10_02.md``.
"""
from __future__ import annotations

import numpy as np

from ._curvemap import (
    Arc,
    EllipseArc,
    RefinedMap,
    Sinusoid,
    TransfiniteMap,
)
from .twod_staggered import _STAG_MIN_SEG_FRAC

__all__ = ["Circle", "Ellipse", "FilletRect", "Rect", "Shape2D",
           "SinusoidalWall", "compile_shapes"]

_DEG = np.pi / 180.0
#: Two walls closer than this (relative to the period) are ONE wall: the
#: shapes' own arithmetic (e.g. ``cx + w / 2`` of two abutting rectangles)
#: differs at round-off, and a 1e-16 segment would otherwise be a sliver.
#: Twelve decades below any physical feature and four above round-off.
_WALL_SNAP = 1e-12
#: Two claims on one grid vertex (or on one edge, sampled) must agree to this,
#: relative to the period -- the same tolerance as the transfinite map's own
#: endpoint check (``_curvemap._EDGE_END_TOL``), so a merge that passes here
#: cannot tear the map there.
_CLAIM_TOL = 1e-12
#: Two outlines CROSS when the boundary of one has points on both sides of the
#: other by more than this (relative to the period).  Shared or touching
#: outlines read ~1e-16; a real crossing is a finite fraction of a feature.
_CROSS_TOL = 1e-9
#: A vertex claim this close to its merged grid vertex (relative to the
#: period) is the grid vertex itself.  It equals the wall snap: walls within
#: _WALL_SNAP are merged, so a claim on the discarded wall sits up to that
#: far off the merged vertex (Phase E2 verifier V-E2-D8).
_VERTEX_SNAP = _WALL_SNAP
#: The fillet radius below which the fillet's own segment ``r / sqrt 2``
#: breaks the sliver contract: ``sqrt(2) * _STAG_MIN_SEG_FRAC`` of the period.
_FILLET_MIN_FRAC = float(np.sqrt(2.0)) * _STAG_MIN_SEG_FRAC


# =========================================================================== #
# Layout records (private)
# =========================================================================== #
class _HardEdge:
    """One piece of a shape's outline carried by ONE ``(u, v)`` grid line:
    ``kind`` 'h' (along ``u`` at ``v = fixed``) or 'v' (along ``v`` at
    ``u = fixed``), spanning the running coordinate ``a .. b``; ``curve`` is
    the exact :class:`~lumenairy.elements.pmm._curvemap.EdgeCurve` over the
    whole span (``s = (t - a) / (b - a)``), or ``None`` for a straight side
    between the two endpoint images."""

    __slots__ = ("kind", "fixed", "a", "b", "curve")

    def __init__(self, kind, fixed, a, b, curve=None):
        self.kind = kind
        self.fixed = float(fixed)
        self.a = float(a)
        self.b = float(b)
        self.curve = curve


class _Layout:
    """A shape's own wall grid: interior ``u`` / ``v`` walls, the EXACT image
    of every endpoint of every hard edge (``vertices[(u, v)] = (x, y)``), the
    hard edges, and the ``(u, v)`` rectangles the shape fills."""

    __slots__ = ("u_walls", "v_walls", "vertices", "edges", "fill")

    def __init__(self, u_walls, v_walls, vertices, edges, fill):
        self.u_walls = [float(w) for w in u_walls]
        self.v_walls = [float(w) for w in v_walls]
        self.vertices = {(float(u), float(v)): np.asarray(xy, dtype=float)
                         for (u, v), xy in vertices.items()}
        self.edges = list(edges)
        self.fill = [tuple(float(c) for c in f) for f in fill]


def _as_eps(eps, who):
    """A scalar (complex) or a ``(3, 3)`` permittivity, validated."""
    if eps is None:
        raise ValueError(f"{who}: eps is required (the shape's permittivity, "
                         f"PUBLIC convention Im(eps) > 0 for loss).")
    a = np.asarray(eps, dtype=complex)
    if a.ndim == 0:
        if not np.isfinite(a):
            raise ValueError(f"{who}: eps must be finite, got {eps!r}.")
        return complex(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{who}: eps must be a scalar or a finite (3, 3) "
                         f"block-form tensor, got shape {a.shape}.")
    return a.copy()


def _as_mu(mu, who):
    """A scalar (complex, nonzero) or a ``(3, 3)`` block-form permeability,
    validated (an out-of-plane ``mu`` is refused by the solver, naming the
    phase that would carry it)."""
    a = np.asarray(mu, dtype=complex)
    if a.ndim == 0:
        if not np.isfinite(a) or a == 0:
            raise ValueError(f"{who}: mu must be finite and nonzero, got "
                             f"{mu!r}.")
        return complex(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{who}: mu must be a scalar or a finite (3, 3) "
                         f"block-form tensor, got shape {a.shape}.")
    return a.copy()


def _fmt(v):
    return f"{float(v):.6g}"


# =========================================================================== #
# The primitives
# =========================================================================== #
class Shape2D:
    """Base class of the shape primitives.  A shape knows its physical
    outline (:meth:`signed_distance`, :meth:`boundary_points`,
    :meth:`area`, :meth:`perimeter`), its material -- ``eps`` (a scalar or a
    ``(3, 3)`` block-form tensor, PUBLIC ``Im(eps) > 0`` for loss) and the
    optional relative permeability ``mu`` (``None`` = 1; a scalar or a
    ``(3, 3)`` block-form tensor) -- and its own wall layout for a given
    period (private; :func:`compile_shapes` reads it).  Every primitive
    takes ``mu=`` as a keyword.  ``name`` labels the shape in every error
    message."""

    #: True when the outline contains a curve (or a moved vertex), so the
    #: shape needs a non-identity map.
    curved = True

    def __init__(self, eps, name, mu=None):
        self.eps = _as_eps(eps, type(self).__name__)
        # Phase D: an optional relative PERMEABILITY (a scalar or a (3, 3)
        # block-form tensor), painted with the shape like its eps
        self.mu = None if mu is None else _as_mu(mu, type(self).__name__)
        self._name = name

    def __setattr__(self, key, value):
        # IMMUTABLE once built: a stack compiles its shapes into the merged
        # wall grid and map at add_layer, so a shape changed afterwards would
        # leave that compiled map silently STALE.  Build a new shape instead.
        if self.__dict__.get("_frozen", False):
            raise AttributeError(
                f"{self.name}: shape primitives are immutable (a stack has "
                f"already compiled this one into its wall grid and map); "
                f"build a new {type(self).__name__} instead of setting "
                f"{key!r}.")
        object.__setattr__(self, key, value)

    def _freeze(self):
        object.__setattr__(self, "_frozen", True)

    @property
    def name(self):
        return self._name or self._default_name()

    def _default_name(self):  # pragma: no cover - protocol
        return type(self).__name__

    def __repr__(self):
        return self.name

    @property
    def is_tensor(self):
        return np.ndim(self.eps) == 2

    @property
    def is_magnetic(self):
        """True when the shape carries a permeability ``mu``."""
        return self.mu is not None

    def bbox(self):  # pragma: no cover - protocol
        """``(x0, x1, y0, y1)`` -- the physical bounding box."""
        raise NotImplementedError

    def signed_distance(self, x, y):  # pragma: no cover - protocol
        """Negative inside the shape, positive outside, zero on the outline
        (exact distance for the closed shapes, a distance-like value for the
        sinusoid)."""
        raise NotImplementedError

    def contains(self, x, y):
        """True strictly inside the outline (the painted region)."""
        return np.asarray(self.signed_distance(x, y)) < 0.0

    def boundary_points(self, n=256):  # pragma: no cover - protocol
        """``(k, 2)`` physical points ON the outline (the material boundary
        the map must carry exactly)."""
        raise NotImplementedError

    def area(self):  # pragma: no cover - protocol
        """The exact area of the painted region (m^2)."""
        raise NotImplementedError

    def perimeter(self):  # pragma: no cover - protocol
        """The exact length of the outline (m)."""
        raise NotImplementedError

    def _layout(self, px, py):  # pragma: no cover - protocol
        raise NotImplementedError

    def _check_inside(self, px, py, strict):
        x0, x1, y0, y1 = self.bbox()
        mx = (_STAG_MIN_SEG_FRAC * px) if strict else -1e-12 * px
        my = (_STAG_MIN_SEG_FRAC * py) if strict else -1e-12 * py
        if not (x0 >= mx and y0 >= my and x1 <= px - mx and y1 <= py - my):
            how = ("at least the sliver width 1e-3 of the period from the "
                   "cell edges" if strict else "inside")
            raise ValueError(
                f"{self.name}: the shape must lie {how} the unit cell "
                f"[0, {px!r}] x [0, {py!r}] (shapes do not wrap across the "
                f"periodic seam); its bounding box is x in [{_fmt(x0)}, "
                f"{_fmt(x1)}], y in [{_fmt(y0)}, {_fmt(y1)}].  Shift the "
                f"lattice origin so the feature sits inside one cell.")


class Rect(Shape2D):
    """An axis-aligned RECTANGLE of width ``w`` (along x) and height ``h``
    (along y) centred at ``(cx, cy)`` -- a sharp-cornered pillar, or a stripe
    when it spans the whole period.

    No map is needed for a rectangle: its four sides are straight walls at
    their physical positions.  A stack (or :func:`compile_shapes` call) made
    of rectangles only runs the shipped UNMAPPED solver on those walls (no
    Jacobian to compose a tensor with), where any tensor ``eps`` -- block-form
    or out-of-plane -- and any ``mu=`` are accepted.

    Example -- a 400 nm square silicon-nitride pillar in a 1.2 um cell::

        from lumenairy.elements.pmm import Rect, pmm_jones_2d_staggered
        pillar = Rect(0.6e-6, 0.6e-6, 0.4e-6, 0.4e-6, eps=2.0 ** 2)
        orders, R, T, J = pmm_jones_2d_staggered(
            1.2e-6, 1.2e-6, None, 1.45, 1.0, 0.5e-6, 1.0e-6,
            shapes=[pillar], background_eps=1.0, n_modes=7)
    """

    curved = False

    def __init__(self, cx, cy, w, h, eps, *, mu=None, name=None):
        super().__init__(eps, name, mu)
        self.cx, self.cy = float(cx), float(cy)
        self.w, self.h = float(w), float(h)
        if not (self.w > 0.0 and self.h > 0.0):
            raise ValueError(f"{self.name}: w and h must be > 0.")
        self._freeze()

    def _default_name(self):
        return (f"Rect(cx={_fmt(self.cx)}, cy={_fmt(self.cy)}, "
                f"w={_fmt(self.w)}, h={_fmt(self.h)})")

    def bbox(self):
        return (self.cx - self.w / 2, self.cx + self.w / 2,
                self.cy - self.h / 2, self.cy + self.h / 2)

    def signed_distance(self, x, y):
        return np.maximum(np.abs(np.asarray(x) - self.cx) - self.w / 2,
                          np.abs(np.asarray(y) - self.cy) - self.h / 2)

    def boundary_points(self, n=256):
        x0, x1, y0, y1 = self.bbox()
        t = np.linspace(0.0, 1.0, max(2, n // 4), endpoint=False)
        return np.concatenate([
            np.stack([x0 + t * (x1 - x0), np.full_like(t, y0)], 1),
            np.stack([np.full_like(t, x1), y0 + t * (y1 - y0)], 1),
            np.stack([x1 - t * (x1 - x0), np.full_like(t, y1)], 1),
            np.stack([np.full_like(t, x0), y1 - t * (y1 - y0)], 1)])

    def area(self):
        return self.w * self.h

    def perimeter(self):
        return 2.0 * (self.w + self.h)

    def _layout(self, px, py):
        self._check_inside(px, py, strict=False)
        x0, x1, y0, y1 = self.bbox()
        # a side ON the cell boundary is the cell boundary itself
        x0 = 0.0 if abs(x0) <= _WALL_SNAP * px else x0
        y0 = 0.0 if abs(y0) <= _WALL_SNAP * py else y0
        x1 = px if abs(x1 - px) <= _WALL_SNAP * px else x1
        y1 = py if abs(y1 - py) <= _WALL_SNAP * py else y1
        verts = {(x0, y0): (x0, y0), (x1, y0): (x1, y0),
                 (x0, y1): (x0, y1), (x1, y1): (x1, y1)}
        edges = [_HardEdge("h", y0, x0, x1), _HardEdge("h", y1, x0, x1),
                 _HardEdge("v", x0, y0, y1), _HardEdge("v", x1, y0, y1)]
        uw = [w for w in (x0, x1) if 0.0 < w < px]
        vw = [w for w in (y0, y1) if 0.0 < w < py]
        return _Layout(uw, vw, verts, edges, [(x0, x1, y0, y1)])


class FilletRect(Shape2D):
    """A rectangle (width ``w``, height ``h``, centre ``(cx, cy)``) whose four
    corners are ROUNDED to the radius ``r`` -- the shape a lithographic
    pillar actually has.

    The layout is the 5 x 5 wall grid the Phase B gates were measured on:
    walls through each fillet's 45-degree point and through its two
    TANGENCY points (where the arc meets the straight sides, and the
    curvature jumps), so every cell edge is a pure arc or a pure straight
    line and the map is analytic inside every cell.

    ``r = 0`` is the sharp-cornered :class:`Rect` (same walls, no map).  A
    radius ``0 < r < sqrt(2) x 1e-3`` of the period (1.4142e-3) is REFUSED: the fillet's own
    segment is ``r / sqrt 2`` wide, and the staggered solver's sliver contract
    forbids segments narrower than 1e-3 of the period
    (``twod_staggered._STAG_MIN_SEG_FRAC``) -- at that size the rounding is
    below what the solver can resolve, and a sharp corner is the faithful
    model.  ``r`` must be below ``min(w, h) / 2`` (a full half-width rounding
    is a :class:`Circle` or a stadium).

    Geometry fidelity, not speed: a fillet changes the device (T00 moves by
    7.4e-3 at ``r`` = 0.2 of the side, planning section 3.4) but the
    efficiencies stay rim-capped near 1e-5 per rung, as for a sharp pillar.

    Example -- the 600 nm square pillar of the planning fixture with 60 nm
    corner radii::

        from lumenairy.elements.pmm import FilletRect, PMM2DStackPure
        st = PMM2DStackPure(1.2e-6, n_substrate=1.45, n_modes=6)
        st.add_layer(0.5e-6, shapes=[FilletRect(0.6e-6, 0.6e-6, 0.6e-6,
                                                0.6e-6, 0.06e-6, eps=4.0)],
                     background_eps=1.0)
        st.set_source(1.0e-6)
        orders, R, T, J = st.solve()
    """

    def __init__(self, cx, cy, w, h, r, eps, *, mu=None, name=None):
        super().__init__(eps, name, mu)
        self.cx, self.cy = float(cx), float(cy)
        self.w, self.h = float(w), float(h)
        self.r = float(r)
        if not (self.w > 0.0 and self.h > 0.0):
            raise ValueError(f"{self.name}: w and h must be > 0.")
        if not self.r >= 0.0:
            raise ValueError(f"{self.name}: the fillet radius must be >= 0.")
        if self.r >= 0.5 * min(self.w, self.h) * (1.0 - 1e-9):
            raise ValueError(
                f"{self.name}: the fillet radius r = {self.r!r} must be below "
                f"min(w, h) / 2 = {0.5 * min(self.w, self.h)!r} (the straight "
                f"sides would vanish; a fully rounded square is a Circle).")
        self.curved = self.r > 0.0
        self._freeze()

    def _default_name(self):
        return (f"FilletRect(cx={_fmt(self.cx)}, cy={_fmt(self.cy)}, "
                f"w={_fmt(self.w)}, h={_fmt(self.h)}, r={_fmt(self.r)})")

    def bbox(self):
        return (self.cx - self.w / 2, self.cx + self.w / 2,
                self.cy - self.h / 2, self.cy + self.h / 2)

    def signed_distance(self, x, y):
        qx = np.abs(np.asarray(x) - self.cx) - (self.w / 2 - self.r)
        qy = np.abs(np.asarray(y) - self.cy) - (self.h / 2 - self.r)
        return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
                + np.minimum(np.maximum(qx, qy), 0.0) - self.r)

    def boundary_points(self, n=256):
        r = self.r
        if r == 0.0:
            return Rect(self.cx, self.cy, self.w, self.h, 1.0).boundary_points(n)
        hx, hy = self.w / 2 - r, self.h / 2 - r
        m = max(2, n // 8)
        t = np.linspace(0.0, 1.0, m, endpoint=False)
        out = []
        for k, (sx, sy) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):
            th = (-90.0 + 90.0 * k) * _DEG + t * 90.0 * _DEG
            c = (self.cx + sx * hx, self.cy + sy * hy)
            out.append(np.stack([c[0] + r * np.cos(th),
                                 c[1] + r * np.sin(th)], 1))
        out.append(np.stack([self.cx - hx + 2 * hx * t,
                             np.full_like(t, self.cy - self.h / 2)], 1))
        out.append(np.stack([self.cx - hx + 2 * hx * t,
                             np.full_like(t, self.cy + self.h / 2)], 1))
        out.append(np.stack([np.full_like(t, self.cx - self.w / 2),
                             self.cy - hy + 2 * hy * t], 1))
        out.append(np.stack([np.full_like(t, self.cx + self.w / 2),
                             self.cy - hy + 2 * hy * t], 1))
        return np.concatenate(out)

    def area(self):
        return self.w * self.h - (4.0 - np.pi) * self.r ** 2

    def perimeter(self):
        return 2.0 * (self.w + self.h) - 8.0 * self.r + 2.0 * np.pi * self.r

    def _layout(self, px, py):
        if self.r == 0.0:
            return Rect(self.cx, self.cy, self.w, self.h, self.eps,
                        mu=self.mu, name=self.name)._layout(px, py)
        rmin = _FILLET_MIN_FRAC * max(px, py)
        if self.r < rmin * (1.0 - 1e-9):
            raise ValueError(
                f"{self.name}: the fillet radius r = {self.r!r} is below "
                f"sqrt(2) x 1e-3 (1.4142e-3) of the period ({rmin!r}).  The "
                f"fillet's "
                f"own wall "
                f"segment is r / sqrt 2 = {float(self.r / np.sqrt(2.0))!r} wide, and "
                f"the staggered solver's SLIVER CONTRACT refuses any segment "
                f"narrower than 1e-3 of the period "
                f"(twod_staggered._STAG_MIN_SEG_FRAC): a narrow segment "
                f"carries spurious modal wavenumbers ~ 1 / width.  Use "
                f"radius=0 (a sharp corner -- the faithful model of a "
                f"rounding this small) or a radius >= {rmin!r}.")
        self._check_inside(px, py, strict=True)
        r = self.r
        cx, cy = self.cx, self.cy
        # Phase B's _fillet_map_5x5, generalised to (cx, cy, w, h): the far
        # walls are MIRRORED as 2 c - wall, which for a centred pillar is
        # the planning probe's "P - wall" to the bit (2 * (P / 2) == P)
        lox, hix = cx - self.w / 2, cx + self.w / 2
        loy, hiy = cy - self.h / 2, cy + self.h / 2
        ax = lox + r * (1.0 - 1.0 / np.sqrt(2.0))
        bx_ = lox + r
        ay = loy + r * (1.0 - 1.0 / np.sqrt(2.0))
        by_ = loy + r
        u = [ax, bx_, 2 * cx - bx_, 2 * cx - ax]
        v = [ay, by_, 2 * cy - by_, 2 * cy - ay]
        verts = {}
        for i in (0, 3):
            for j in (0, 3):
                verts[(u[i], v[j])] = (u[i], v[j])          # 45-degree points
        for i in (1, 2):
            verts[(u[i], v[0])] = (u[i], loy)               # bottom tangency
            verts[(u[i], v[3])] = (u[i], hiy)               # top tangency
        for j in (1, 2):
            verts[(u[0], v[j])] = (lox, v[j])               # left tangency
            verts[(u[3], v[j])] = (hix, v[j])               # right tangency
        cBL, cBR = (bx_, by_), (2 * cx - bx_, by_)
        cTL, cTR = (bx_, 2 * cy - by_), (2 * cx - bx_, 2 * cy - by_)
        edges = [
            _HardEdge("h", v[0], u[0], u[1], Arc(cBL, r, 225 * _DEG,
                                                 270 * _DEG)),
            _HardEdge("h", v[0], u[2], u[3], Arc(cBR, r, 270 * _DEG,
                                                 315 * _DEG)),
            _HardEdge("h", v[3], u[0], u[1], Arc(cTL, r, 135 * _DEG,
                                                 90 * _DEG)),
            _HardEdge("h", v[3], u[2], u[3], Arc(cTR, r, 90 * _DEG,
                                                 45 * _DEG)),
            _HardEdge("v", u[0], v[0], v[1], Arc(cBL, r, 225 * _DEG,
                                                 180 * _DEG)),
            _HardEdge("v", u[0], v[2], v[3], Arc(cTL, r, 180 * _DEG,
                                                 135 * _DEG)),
            _HardEdge("v", u[3], v[0], v[1], Arc(cBR, r, 315 * _DEG,
                                                 360 * _DEG)),
            _HardEdge("v", u[3], v[2], v[3], Arc(cTR, r, 0 * _DEG,
                                                 45 * _DEG)),
            # the four flat sides: straight between the tangency images
            _HardEdge("h", v[0], u[1], u[2]), _HardEdge("h", v[3], u[1], u[2]),
            _HardEdge("v", u[0], v[1], v[2]), _HardEdge("v", u[3], v[1], v[2]),
        ]
        return _Layout(u, v, verts, edges, [(u[0], u[3], v[0], v[3])])


class Circle(Shape2D):
    """A circular DISK of radius ``r`` centred at ``(cx, cy)``.

    The default layout is the 3 x 3 wall grid of the Phase B gates: walls
    through the circle's four 45-degree points (``c -+ r / sqrt 2``), every
    grid vertex at its own physical position, and the middle cell's four
    edges 90-degree arcs -- so the middle cell IS the disk, exactly (its
    mapped area is ``pi r^2`` to round-off), and the circle costs the same
    wall grid as a square pillar.  This layout landed on an independent 3-D
    finite-element oracle to 1.9e-6 per order (Phase B gate B3).

    ``core=f`` (``0 < f < 1``) selects the 5 x 5 layout instead: the disk is
    the inner 3 x 3 block, its centre a straight-edged square of half-size
    ``f r / sqrt 2``.  It costs 2.8x the pencil at equal ``n_modes`` but gives
    the disk's interior a better-shaped parametrisation: its in-plane modes
    converge faster, and at OBLIQUE incidence a uniform film under it is four
    to six decades closer at equal ``n_modes`` (Phase B finding F-B6) --
    prefer it for oblique / conical work.

    Example -- a 360 nm-radius pillar (n = 2) in a 1.2 um cell, as a single
    layer and as the circle of the Phase B gates::

        from lumenairy.elements.pmm import Circle, pmm_jones_2d_staggered
        disk = Circle(0.6e-6, 0.6e-6, 0.36e-6, eps=4.0)
        orders, R, T, J = pmm_jones_2d_staggered(
            1.2e-6, 1.2e-6, None, 1.45, 1.0, 0.5e-6, 1.0e-6,
            shapes=[disk], background_eps=1.0, n_modes=8)
    """

    def __init__(self, cx, cy, r, eps, *, core=None, mu=None, name=None):
        super().__init__(eps, name, mu)
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)
        if not self.r > 0.0:
            raise ValueError(f"{self.name}: r must be > 0.")
        if core is not None and not 0.0 < float(core) < 1.0:
            raise ValueError(f"{self.name}: core must be None (3 x 3 layout) "
                             f"or a fraction in (0, 1) (5 x 5 layout), got "
                             f"{core!r}.")
        self.core = None if core is None else float(core)
        self._freeze()

    def _default_name(self):
        return (f"Circle(cx={_fmt(self.cx)}, cy={_fmt(self.cy)}, "
                f"r={_fmt(self.r)})")

    def bbox(self):
        return (self.cx - self.r, self.cx + self.r, self.cy - self.r,
                self.cy + self.r)

    def signed_distance(self, x, y):
        return np.hypot(np.asarray(x) - self.cx,
                        np.asarray(y) - self.cy) - self.r

    def boundary_points(self, n=256):
        th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([self.cx + self.r * np.cos(th),
                         self.cy + self.r * np.sin(th)], 1)

    def area(self):
        return np.pi * self.r ** 2

    def perimeter(self):
        return 2.0 * np.pi * self.r

    def _layout(self, px, py):
        self._check_inside(px, py, strict=True)
        r = self.r
        cx, cy = self.cx, self.cy
        if self.core is None:
            # Phase B's _circle_map_3x3, to the bit
            h = r / np.sqrt(2.0)
            c = (cx, cy)
            u = [c[0] - h, c[0] + h]
            v = [c[1] - h, c[1] + h]
            verts = {(uu, vv): (uu, vv) for uu in u for vv in v}
            edges = [
                _HardEdge("h", v[0], u[0], u[1],
                          Arc(c, r, 225 * _DEG, 315 * _DEG)),
                _HardEdge("h", v[1], u[0], u[1],
                          Arc(c, r, 135 * _DEG, 45 * _DEG)),
                _HardEdge("v", u[0], v[0], v[1],
                          Arc(c, r, 225 * _DEG, 135 * _DEG)),
                _HardEdge("v", u[1], v[0], v[1],
                          Arc(c, r, -45 * _DEG, 45 * _DEG)),
            ]
            return _Layout(u, v, verts, edges, [(u[0], u[1], v[0], v[1])])
        # Phase B's _circle_map_5x5 (inner = core), generalised to (cx, cy)
        c5 = np.array([cx, cy])
        a1x = cx - r / np.sqrt(2.0)
        a1y = cy - r / np.sqrt(2.0)
        hin = self.core * r / np.sqrt(2.0)
        a2x, a2y = cx - hin, cy - hin
        u = [a1x, a2x, 2 * cx - a2x, 2 * cx - a1x]
        v = [a1y, a2y, 2 * cy - a2y, 2 * cy - a1y]
        phi = np.arcsin(hin / r)

        def circ(th):
            return (c5[0] + r * np.cos(th), c5[1] + r * np.sin(th))

        # loop vertices: local index 1..4 of the 6-wall grid -> u[k - 1]
        def img(i, j):
            if i in (1, 4) and j in (1, 4):
                return (u[i - 1], v[j - 1])            # 45-degree points
            if j in (1, 4) and i in (2, 3):            # bottom / top side
                sgn = -1 if i == 2 else 1
                base = 270 * _DEG if j == 1 else 90 * _DEG
                return circ(base + (sgn * phi if j == 1 else -sgn * phi))
            sgn = -1 if j == 2 else 1                  # left / right side
            return circ(180 * _DEG - sgn * phi if i == 1 else sgn * phi)

        loop = [(i, j) for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)
                if i in (1, 4) or j in (1, 4)]
        im = {(i, j): np.array(img(i, j), dtype=np.float64)
              for i, j in loop}
        verts5 = {(u[i - 1], v[j - 1]): im[(i, j)] for i, j in loop}
        edges = []
        for i in (1, 2, 3):
            for j in (1, 4):
                edges.append(_HardEdge(
                    "h", v[j - 1], u[i - 1], u[i],
                    Arc.through(im[(i, j)], im[(i + 1, j)], c5)))
        for j in (1, 2, 3):
            for i in (1, 4):
                edges.append(_HardEdge(
                    "v", u[i - 1], v[j - 1], v[j],
                    Arc.through(im[(i, j)], im[(i, j + 1)], c5)))
        return _Layout(u, v, verts5, edges, [(u[0], u[3], v[0], v[3])])


class Ellipse(Shape2D):
    """An elliptical DISK with semi-axes ``a`` (along x before rotation) and
    ``b`` centred at ``(cx, cy)``, rotated counter-clockwise by ``angle``
    (radians, ``|angle| < pi / 4``; a quarter-turn is a swap of ``a`` and
    ``b``).

    Layout: 3 x 3, the middle cell the disk, its corners the points where
    the outline's outward normal points at 45, 135, 225, 315 degrees, and its
    edges :class:`~lumenairy.elements.pmm._curvemap.EllipseArc`
    s.  Axis-aligned (``angle = 0``), those corners are the corners of an
    axis-aligned rectangle and sit at their own physical positions (the
    Phase B gate layout, to the bit).  Rotated, they are not, so the four
    corner vertices are MOVED onto the ellipse (the walls pass through the
    midpoints of the corner pairs); the outline is still exact, the cells
    around the disk carry a bilinear shear.

    Example -- an elliptical post 400 nm x 280 nm (semi-axes 200 / 140 nm),
    tilted by 20 degrees::

        import numpy as np
        from lumenairy.elements.pmm import Ellipse, compile_shapes
        post = Ellipse(0.6e-6, 0.6e-6, 0.2e-6, 0.14e-6, eps=4.0,
                       angle=np.deg2rad(20.0))
        eps_cell, xw, yw, cmap = compile_shapes(1.2e-6, 1.2e-6, [post], 1.0)
    """

    def __init__(self, cx, cy, a, b, eps, *, angle=0.0, mu=None,
                 name=None):
        super().__init__(eps, name, mu)
        self.cx, self.cy = float(cx), float(cy)
        self.a, self.b = float(a), float(b)
        self.angle = float(angle)
        if not (self.a > 0.0 and self.b > 0.0):
            raise ValueError(f"{self.name}: the semi-axes must be > 0.")
        if not abs(self.angle) < np.pi / 4:
            raise ValueError(
                f"{self.name}: |angle| must be < pi / 4 (45 degrees); rotate "
                f"by a quarter-turn by swapping a and b instead.")
        self._freeze()

    def _default_name(self):
        s = (f"Ellipse(cx={_fmt(self.cx)}, cy={_fmt(self.cy)}, "
             f"a={_fmt(self.a)}, b={_fmt(self.b)}")
        if self.angle:
            s += f", angle={_fmt(self.angle)}"
        return s + ")"

    def _pt(self, t):
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        X, Y = self.a * np.cos(t), self.b * np.sin(t)
        return self.cx + ca * X - sa * Y, self.cy + sa * X + ca * Y

    def bbox(self):
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        ex = np.hypot(self.a * ca, self.b * sa)
        ey = np.hypot(self.a * sa, self.b * ca)
        return (self.cx - ex, self.cx + ex, self.cy - ey, self.cy + ey)

    def signed_distance(self, x, y):
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        dx, dy = np.asarray(x) - self.cx, np.asarray(y) - self.cy
        X = ca * dx + sa * dy
        Y = -sa * dx + ca * dy
        return (np.hypot(X / self.a, Y / self.b) - 1.0) * min(self.a, self.b)

    def boundary_points(self, n=256):
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        x, y = self._pt(t)
        return np.stack([x, y], 1)

    def area(self):
        return np.pi * self.a * self.b

    def perimeter(self):
        from scipy.special import ellipe
        A, B = max(self.a, self.b), min(self.a, self.b)
        return 4.0 * A * float(ellipe(1.0 - (B / A) ** 2))

    def _layout(self, px, py):
        self._check_inside(px, py, strict=True)
        a, b = self.a, self.b
        if self.angle == 0.0:
            # Phase B's _ellipse_map_3x3, to the bit
            c = (self.cx, self.cy)
            hu, hv = a / np.sqrt(2.0), b / np.sqrt(2.0)
            u = [c[0] - hu, c[0] + hu]
            v = [c[1] - hv, c[1] + hv]
            verts = {(uu, vv): (uu, vv) for uu in u for vv in v}
            ax = (a, b)
            edges = [
                _HardEdge("h", v[0], u[0], u[1],
                          EllipseArc(c, ax, 225 * _DEG, 315 * _DEG)),
                _HardEdge("h", v[1], u[0], u[1],
                          EllipseArc(c, ax, 135 * _DEG, 45 * _DEG)),
                _HardEdge("v", u[0], v[0], v[1],
                          EllipseArc(c, ax, 225 * _DEG, 135 * _DEG)),
                _HardEdge("v", u[1], v[0], v[1],
                          EllipseArc(c, ax, -45 * _DEG, 45 * _DEG)),
            ]
            return _Layout(u, v, verts, edges, [(u[0], u[1], v[0], v[1])])
        c = (self.cx, self.cy)
        al = self.angle

        # the corners where the OUTWARD NORMAL points at 225 / 315 / 45 /
        # 135 degrees in the lab (the analogue of the circle's 45-degree
        # points): parametric t = atan2(b sin(psi - angle), a cos(psi -
        # angle)).  The parametric 45-degree points fold the side cells for
        # aspect >= 1.5 at 30 deg, 2 at 20 deg, 5 at 10 deg (Phase C
        # verifier V-D2).
        def tn(psi):
            p = psi * _DEG - al
            return float(np.arctan2(b * np.sin(p), a * np.cos(p)))
        tBL = tn(225.0) % (2 * np.pi)
        tBR = tBL + (tn(315.0) - tBL) % (2 * np.pi)
        tTR = tBR + (tn(45.0) - tBR) % (2 * np.pi)
        tTL = tTR + (tn(135.0) - tTR) % (2 * np.pi)
        P = {k: np.array(self._pt(t)) for k, t in
             (("BL", tBL), ("BR", tBR), ("TR", tTR), ("TL", tTL))}
        u = [0.5 * (P["BL"][0] + P["TL"][0]), 0.5 * (P["BR"][0] + P["TR"][0])]
        v = [0.5 * (P["BL"][1] + P["BR"][1]), 0.5 * (P["TL"][1] + P["TR"][1])]
        vrot = {(u[0], v[0]): P["BL"], (u[1], v[0]): P["BR"],
                (u[1], v[1]): P["TR"], (u[0], v[1]): P["TL"]}
        ax = (a, b)
        edges = [
            _HardEdge("h", v[0], u[0], u[1],
                      EllipseArc(c, ax, tBL, tBR, angle=al)),
            _HardEdge("h", v[1], u[0], u[1],
                      EllipseArc(c, ax, tTL, tTR, angle=al)),
            _HardEdge("v", u[0], v[0], v[1],
                      EllipseArc(c, ax, tBL + 2 * np.pi, tTL, angle=al)),
            _HardEdge("v", u[1], v[0], v[1],
                      EllipseArc(c, ax, tBR, tTR, angle=al)),
        ]
        return _Layout(u, v, vrot, edges, [(u[0], u[1], v[0], v[1])])


class SinusoidalWall(Shape2D):
    """An interface that wiggles sinusoidally across the whole cell.

    ``axis='x'``: the wall sits at ``x = x0 + amplitude sin(2 pi n y / p_y +
    phase)`` and runs across the cell in ``y`` (``n = period_count``, a
    positive integer so the wall is lattice-periodic); ``eps`` fills the side
    ``x > wall`` up to the cell edge ``x = p_x`` -- or, with ``width=``, the
    RIDGE between the wall and its in-phase copy ``x0 + width``.
    ``axis='y'`` is the same with x and y exchanged (``y = x0 + amplitude
    sin(2 pi n x / p_x + phase)``).  Painting order makes a single-wall
    interface between two materials: ``[SinusoidalWall(..., eps=e2)]`` on
    ``background_eps=e1``.

    Layout: the wall IS the grid line ``u = x0`` (``v = x0`` for
    ``axis='y'``); its vertices slide along the wall (``x0 + A sin(...)`` at
    each crossing wall), so the cells on its two sides are blended between
    the sinusoid and the straight walls next to them.  ``x0 -+ |amplitude|``
    (and ``x0 + width -+ |amplitude|``) must lie inside the cell.

    Example -- a 600 nm-wide sinusoidal ridge, amplitude 120 nm, one period
    of wiggle per cell::

        from lumenairy.elements.pmm import SinusoidalWall, compile_shapes
        ridge = SinusoidalWall("x", 0.3e-6, 0.12e-6, eps=4.0, width=0.6e-6)
        eps_cell, xw, yw, cmap = compile_shapes(1.2e-6, 1.2e-6, [ridge], 1.0)
    """

    def __init__(self, axis, x0, amplitude, period_count=1, phase=0.0, *,
                 eps, width=None, mu=None, name=None):
        super().__init__(eps, name, mu)
        if axis not in ("x", "y"):
            raise ValueError(f"SinusoidalWall: axis must be 'x' (the wall "
                             f"position is an x, running along y) or 'y', "
                             f"got {axis!r}.")
        self.axis = axis
        self.x0 = float(x0)
        self.amplitude = float(amplitude)
        n = int(period_count)
        if n != period_count or n < 1:
            raise ValueError(
                f"{self.name}: period_count must be a positive integer (the "
                f"wall must repeat with the lattice), got {period_count!r}.")
        self.period_count = n
        self.phase = float(phase)
        self.width = None if width is None else float(width)
        if self.width is not None and not self.width > 0.0:
            raise ValueError(f"{self.name}: width must be > 0.")
        self.curved = self.amplitude != 0.0
        self._periods = None          # (p_run, p_pos), set by _layout
        self._freeze()

    def _default_name(self):
        s = (f"SinusoidalWall(axis={self.axis!r}, x0={_fmt(self.x0)}, "
             f"amplitude={_fmt(self.amplitude)}")
        if self.period_count != 1:
            s += f", period_count={self.period_count}"
        if self.width is not None:
            s += f", width={_fmt(self.width)}"
        return s + ")"

    def _need_periods(self):
        if self._periods is None:
            raise ValueError(f"{self.name}: the cell period is not known "
                             f"yet -- compile the shape first.")
        return self._periods

    def _wall(self, t, base):
        p_run, _ = self._need_periods()
        k = 2.0 * np.pi * self.period_count / p_run
        return base + self.amplitude * np.sin(k * np.asarray(t) + self.phase)

    def _bases(self):
        return [self.x0] + ([] if self.width is None
                            else [self.x0 + self.width])

    def bbox(self):
        p_run, p_pos = self._need_periods()
        lo = self.x0 - abs(self.amplitude)
        hi = (p_pos if self.width is None
              else self.x0 + self.width + abs(self.amplitude))
        if self.axis == "x":
            return (lo, hi, 0.0, p_run)
        return (0.0, p_run, lo, hi)

    def signed_distance(self, x, y):
        pos, run = (x, y) if self.axis == "x" else (y, x)
        pos = np.asarray(pos)
        d = self._wall(run, self.x0) - pos
        if self.width is not None:
            d = np.maximum(d, pos - self._wall(run, self.x0 + self.width))
        return d

    def boundary_points(self, n=256):
        p_run, _ = self._need_periods()
        t = np.linspace(0.0, p_run, n, endpoint=False)
        out = []
        for base in self._bases():
            w = self._wall(t, base)
            out.append(np.stack([w, t], 1) if self.axis == "x"
                       else np.stack([t, w], 1))
        return np.concatenate(out)

    def area(self):
        p_run, p_pos = self._need_periods()
        # the sine integrates to zero over whole periods
        return (self.width if self.width is not None
                else p_pos - self.x0) * p_run

    def perimeter(self):
        p_run, _ = self._need_periods()
        from numpy.polynomial.legendre import leggauss
        k = 2.0 * np.pi * self.period_count / p_run
        xg, wg = leggauss(64)
        tot = 0.0
        for q in range(4 * self.period_count):      # per quarter-wave piece
            t0 = q * p_run / (4 * self.period_count)
            t1 = (q + 1) * p_run / (4 * self.period_count)
            t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * xg
            f = np.sqrt(1.0 + (self.amplitude * k
                               * np.cos(k * t + self.phase)) ** 2)
            tot += 0.5 * (t1 - t0) * float(wg @ f)
        return tot * len(self._bases())

    def _layout(self, px, py):
        p_run, p_pos = (py, px) if self.axis == "x" else (px, py)
        if self._periods not in (None, (p_run, p_pos)):
            raise ValueError(
                f"{self.name}: this wall was compiled for the periods "
                f"{self._periods}; build a new SinusoidalWall for another "
                f"cell.")
        object.__setattr__(self, "_periods", (p_run, p_pos))
        A = abs(self.amplitude)
        for base in self._bases():
            if not (base - A >= _STAG_MIN_SEG_FRAC * p_pos
                    and base + A <= p_pos * (1.0 - _STAG_MIN_SEG_FRAC)):
                raise ValueError(
                    f"{self.name}: the wall {_fmt(base)} -+ |amplitude| "
                    f"{_fmt(A)} must lie inside the cell (0, {p_pos!r}) by at "
                    f"least the sliver width 1e-3 of the period.")
        pos_walls = self._bases()
        verts, edges = {}, []
        for base in pos_walls:
            crv = Sinusoid(base, self.amplitude, p_run / self.period_count,
                           0.0, p_run, phase=self.phase,
                           along=("y" if self.axis == "x" else "x"))
            e0 = crv(np.array([0.0, 1.0]))[0]
            kind = "v" if self.axis == "x" else "h"
            for t, e in ((0.0, e0[0]), (p_run, e0[1])):
                key = (base, t) if self.axis == "x" else (t, base)
                verts[key] = e
            edges.append(_HardEdge(kind, base, 0.0, p_run,
                                   crv if self.curved else None))
        top = p_pos if self.width is None else self.x0 + self.width
        if self.axis == "x":
            fill = [(self.x0, top, 0.0, p_run)]
            return _Layout(pos_walls, [], verts, edges, fill)
        fill = [(0.0, p_run, self.x0, top)]
        return _Layout([], pos_walls, verts, edges, fill)


# =========================================================================== #
# The merge: many shapes (in many layers) -> ONE wall grid and ONE map
# =========================================================================== #
def _curve_piece(e, ta, tb):
    """The sub-curve of hard edge ``e`` between the running coordinates
    ``ta`` and ``tb`` (``None`` for a straight side)."""
    if e.curve is None:
        return None
    if ta == e.a and tb == e.b:
        return e.curve
    if isinstance(e.curve, Sinusoid) and e.curve.t0 == e.a and \
            e.curve.t1 == e.b:
        return e.curve.between(ta, tb)       # its parameter IS the coordinate
    L = e.b - e.a
    return e.curve.piece((ta - e.a) / L, (tb - e.a) / L)


def _curve_point(e, t, ends):
    """The image of the point at running coordinate ``t`` on hard edge
    ``e`` (``ends`` = the exact images of its two endpoints)."""
    s = (t - e.a) / (e.b - e.a)
    if e.curve is None:
        return ends[0] + s * (ends[1] - ends[0])
    if isinstance(e.curve, Sinusoid) and e.curve.t0 == e.a:
        return e.curve.between(t, t + 1.0)(np.array([0.0]))[0][0]
    return e.curve(np.array([s]))[0][0]


class _Grid1D:
    """The merged walls of one axis with their owners."""

    def __init__(self, period, axis):
        self.period = float(period)
        self.axis = axis
        self.vals = [0.0, self.period]
        self.owners = [["the unit-cell edge"], ["the unit-cell edge"]]

    def add(self, w, owner):
        tol = _WALL_SNAP * self.period
        for k, v in enumerate(self.vals):
            if abs(v - w) <= tol:
                if owner not in self.owners[k]:
                    self.owners[k].append(owner)
                return
        self.vals.append(float(w))
        self.owners.append([owner])

    def finish(self):
        order = np.argsort(self.vals, kind="stable")
        self.vals = [self.vals[k] for k in order]
        self.owners = [self.owners[k] for k in order]
        self.b = np.asarray(self.vals, dtype=float)

    def index(self, w):
        k = int(np.argmin(np.abs(self.b - w)))
        if abs(self.b[k] - w) > _WALL_SNAP * self.period:
            raise AssertionError(f"wall {w!r} not on the merged grid")
        return k

    def check_slivers(self):
        d = np.diff(self.b)
        bar = _STAG_MIN_SEG_FRAC * self.period * (1.0 - 1e-9)
        for kk in np.nonzero(d < bar)[0]:
            k = int(kk)
            who = sorted(set(self.owners[k]) | set(self.owners[k + 1]))
            raise ValueError(
                f"compile_shapes: the merged {self.axis}-walls "
                f"{float(self.b[k])!r} and {float(self.b[k + 1])!r} are only "
                f"{float(d[k])!r} apart ({float(d[k]) / self.period:.3g} of "
                f"the period) -- below the staggered solver's SLIVER "
                f"CONTRACT (no segment narrower than 1e-3 of the period, "
                f"twod_staggered._STAG_MIN_SEG_FRAC).  The walls belong to "
                f"{' and '.join(who)}.  Align the two outlines exactly (a "
                f"shared wall is free) or move them apart.")


def _refine(b, n_target):
    """Halve the widest segment until ``n_target`` segments (ties: the
    lowest index) -- the plan's legal h-refinement."""
    b = list(b)
    while len(b) - 1 < n_target:
        d = np.diff(b)
        k = int(np.argmax(d))
        b.insert(k + 1, 0.5 * (b[k] + b[k + 1]))
    return np.asarray(b, dtype=float)


def _crossing(A, B, tol):
    """True when the outline of ``B`` has points strictly on both sides of
    ``A``'s outline (the two outlines cross in plan view)."""
    pts = B.boundary_points(512)
    d = np.asarray(A.signed_distance(pts[:, 0], pts[:, 1]))
    return bool(np.any(d < -tol) and np.any(d > tol))


def _merge(px, py, layers, grid_hint=None):
    """The core of :func:`compile_shapes` for one or many layers.

    ``layers`` is a list of ``(label, shapes, background_eps)`` or
    ``(label, shapes, background_eps, background_mu)``.  Returns
    ``(u_bounds, v_bounds, cmap, cells, identity, mu_cells)`` -- ``cells``
    one ``(N, N)`` (or ``(N, N, 3, 3)``) permittivity grid per layer,
    ``identity`` True when the merged map is the identity (no curve, no moved
    vertex), ``mu_cells`` one permeability grid per layer (``None`` for a
    layer with no ``mu`` anywhere; a shape without ``mu`` paints ``mu = 1``)."""
    px, py = float(px), float(py)
    scale = max(px, py)
    items = []
    for lab, shapes, *_bg in layers:
        for sh in shapes:
            if not isinstance(sh, Shape2D):
                raise TypeError(
                    f"compile_shapes: every shape must be a shape primitive "
                    f"(Rect, FilletRect, Circle, Ellipse, SinusoidalWall), "
                    f"got {type(sh).__name__}.")
            who = f"{sh.name}" if lab is None else f"{lab}: {sh.name}"
            items.append((who, sh, sh._layout(px, py)))
    # -- outlines that cross in plan view: one map cannot carry both --------
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            wa, A, _ = items[i]
            wb, B, _ = items[j]
            if not (A.curved or B.curved):
                continue                    # straight lines cross at a vertex
            tol = _CROSS_TOL * scale
            if _crossing(A, B, tol) or _crossing(B, A, tol):
                raise ValueError(
                    f"compile_shapes: the outlines of {wa} and {wb} CROSS in "
                    f"plan view.  One coordinate map cannot carry two "
                    f"crossing curves exactly.  Move the shapes apart or nest "
                    f"one inside the other; if the outlines are in DIFFERENT "
                    f"layers, PMM2DStackPure(..., layer_grids='per-layer') "
                    f"joins the layers' own maps by the curved mortar (Phase "
                    f"E2); outlines in ONE layer cannot be split into two "
                    f"layers without changing the device.")
    # -- the merged walls -----------------------------------------------------
    gu, gv = _Grid1D(px, "u"), _Grid1D(py, "v")
    for who, _sh, lay in items:
        for w in lay.u_walls:
            gu.add(w, who)
        for w in lay.v_walls:
            gv.add(w, who)
    gu.finish()
    gv.finish()
    gu.check_slivers()
    gv.check_slivers()
    U1, V1 = gu.b, gv.b
    nx, ny = U1.size - 1, V1.size - 1
    # -- claims on vertices and edges -----------------------------------------
    vclaims: dict[tuple[int, int], list[tuple[np.ndarray, str]]] = {}
    eclaims: dict[tuple[str, int, int], list[tuple[object, str]]] = {}
    for who, _sh, lay in items:
        ends = {}
        for (u, v), xy in lay.vertices.items():
            key = (gu.index(u), gv.index(v))
            ends[key] = xy
            vclaims.setdefault(key, []).append((xy, who))
        for e in lay.edges:
            if e.kind == "h":
                j = gv.index(e.fixed)
                ia, ib = gu.index(e.a), gu.index(e.b)
                run = U1
                pa, pb = ends[(ia, j)], ends[(ib, j)]
            else:
                i = gu.index(e.fixed)
                ia, ib = gv.index(e.a), gv.index(e.b)
                run = V1
                pa, pb = ends[(i, ia)], ends[(i, ib)]
            for k in range(ia, ib):
                ekey = ("h", k, j) if e.kind == "h" else ("v", i, k)
                ta = e.a if k == ia else float(run[k])
                tb = e.b if k + 1 == ib else float(run[k + 1])
                eclaims.setdefault(ekey, []).append(
                    (_curve_piece(e, ta, tb), who))
            for k in range(ia + 1, ib):
                key = (k, j) if e.kind == "h" else (i, k)
                vclaims.setdefault(key, []).append(
                    (_curve_point(e, float(run[k]), (pa, pb)), who))
    V = np.empty((nx + 1, ny + 1, 2))
    V[..., 0] = U1[:, None]
    V[..., 1] = V1[None, :]
    tol = _CLAIM_TOL * scale
    for key, cl in vclaims.items():
        xy0, w0 = cl[0]
        for xy, w in cl[1:]:
            if float(np.max(np.abs(np.asarray(xy) - xy0))) > tol and w != w0:
                raise ValueError(
                    f"compile_shapes: {w0} and {w} need DIFFERENT physical "
                    f"positions for the same grid vertex (near x = "
                    f"{_fmt(xy0[0])}, y = {_fmt(xy0[1])}: "
                    f"{tuple(map(float, xy0))} vs {tuple(map(float, xy))}) "
                    f"-- their outlines meet in one cell in two "
                    f"different ways (e.g. a rounded corner over a sharp one, "
                    f"or an edge laid on another shape's flat side).  One map "
                    f"cannot carry both.  If the shapes are in DIFFERENT "
                    f"layers, layer_grids='per-layer' gives each layer its "
                    f"own map (Phase E2).")
        V[key] = xy0
    # A claim within a few ulps of its grid vertex IS the grid vertex: two
    # shapes' walls are snapped into one (_WALL_SNAP) but their vertex
    # claims keep each shape's own arithmetic (cx - w / 2 vs cy + h / 2), and
    # a straight edge's interior crossing is placed by linear interpolation;
    # both land 1 ulp off the merged wall, which would make a rectangles-only
    # merge a non-identity map (the mapped solver, tensors refused).
    # (Phase C verifier V-D1.)
    G = np.empty_like(V)
    G[..., 0] = U1[:, None]
    G[..., 1] = V1[None, :]
    near = np.max(np.abs(V - G), axis=-1) <= _VERTEX_SNAP * scale
    V[near] = G[near]
    curved = {}
    s5 = np.linspace(0.0, 1.0, 5)
    for ek, ecl in eclaims.items():
        kind, i, j = ek
        P0 = V[i, j]
        P1 = V[i + 1, j] if kind == "h" else V[i, j + 1]

        def samples(c, P0=P0, P1=P1):
            if c is None:
                return P0[None, :] + s5[:, None] * (P1 - P0)[None, :]
            return c(s5)[0]
        c0, w0 = ecl[0]
        ref = samples(c0)
        for c, w in ecl[1:]:
            if float(np.max(np.abs(samples(c) - ref))) > tol:
                raise ValueError(
                    f"compile_shapes: {w0} and {w} place DIFFERENT curves on "
                    f"one cell edge (near x = {_fmt(ref[2, 0])}, y = "
                    f"{_fmt(ref[2, 1])}).  Two layers may share a curve but "
                    f"not lay two different outlines on one grid line; a "
                    f"common map for them is Phase E of the curved-cell plan.")
        cc = next((c for c, _w in ecl if c is not None), None)
        if cc is not None:
            curved[ek] = cc
    tm = TransfiniteMap(U1, V1, V, curved, _validate=False)
    _check_fold(tm, gu, gv, vclaims, eclaims)
    tm.validate(n=12)
    # -- square the grid (Nx == Ny) and honour grid_hint ----------------------
    n_min = 0
    if grid_hint is not None:
        gh = np.atleast_1d(np.asarray(grid_hint, dtype=int)).ravel()
        if gh.size not in (1, 2) or np.any(gh < 1):
            raise ValueError(f"compile_shapes: grid_hint must be an int or a "
                             f"pair of ints >= 1 (the minimum segment count "
                             f"per axis), got {grid_hint!r}.")
        n_min = int(np.max(gh))
    n = max(nx, ny, n_min)
    U, Vb = _refine(U1, n), _refine(V1, n)
    cmap = tm if (U.size == U1.size and Vb.size == V1.size) else \
        RefinedMap(tm, U, Vb)
    identity = (not curved) and bool(np.all(V[..., 0] == U1[:, None])
                                     and np.all(V[..., 1] == V1[None, :]))
    # -- paint every layer on the merged (u, v) cells -------------------------
    uc = 0.5 * (U[:-1] + U[1:])
    vc = 0.5 * (Vb[:-1] + Vb[1:])
    cells = []
    mu_cells = []
    for lab, shapes, bg, *bgm in layers:
        bgv = _as_eps(bg, "compile_shapes: background_eps")
        bgmu = bgm[0] if bgm else None
        mu_cells.append(_paint_mu(shapes, bgmu, px, py, uc, vc))
        tensor = np.ndim(bgv) == 2 or any(sh.is_tensor for sh in shapes)
        N = uc.size
        if tensor:
            cell = np.empty((N, N, 3, 3), dtype=complex)
            cell[:] = bgv if np.ndim(bgv) == 2 else bgv * np.eye(3)
        else:
            cell = np.full((N, N), bgv, dtype=complex)
        for sh in shapes:
            lay = sh._layout(px, py)
            m = np.zeros((N, N), dtype=bool)
            for (u0, u1, v0, v1) in lay.fill:
                m |= ((uc[:, None] > u0) & (uc[:, None] < u1)
                      & (vc[None, :] > v0) & (vc[None, :] < v1))
            if tensor:
                cell[m] = sh.eps if sh.is_tensor else sh.eps * np.eye(3)
            else:
                cell[m] = sh.eps
        cells.append(cell)
    return U, Vb, cmap, cells, identity, mu_cells


def _paint_mu(shapes, background_mu, px, py, uc, vc):
    """The permeability grid of one layer, painted exactly like its eps
    (same cells, same order): ``None`` when neither the background nor any
    shape carries a ``mu``; a shape without ``mu`` paints ``mu = 1``."""
    if background_mu is None and not any(sh.is_magnetic for sh in shapes):
        return None
    bgv = 1.0 + 0.0j if background_mu is None else _as_mu(
        background_mu, "compile_shapes: background_mu")
    mus = [bgv] + [sh.mu if sh.is_magnetic else 1.0 + 0.0j for sh in shapes]
    tensor = any(np.ndim(m) == 2 for m in mus)
    N = uc.size
    if tensor:
        cell = np.empty((N, N, 3, 3), dtype=complex)
        cell[:] = bgv if np.ndim(bgv) == 2 else bgv * np.eye(3)
    else:
        cell = np.full((N, N), bgv, dtype=complex)
    for sh, m in zip(shapes, mus[1:]):
        lay = sh._layout(px, py)
        msk = np.zeros((N, N), dtype=bool)
        for (u0, u1, v0, v1) in lay.fill:
            msk |= ((uc[:, None] > u0) & (uc[:, None] < u1)
                    & (vc[None, :] > v0) & (vc[None, :] < v1))
        if tensor:
            cell[msk] = m if np.ndim(m) == 2 else m * np.eye(3)
        else:
            cell[msk] = m
    return cell


def _check_fold(tm, gu, gv, vclaims, eclaims, n=12):
    """Raise, naming the shapes, if the merged map folds (``det J <= 0``) in
    any cell -- two outlines too close, or a straight wall through another
    shape's curved transition cell."""
    from numpy.polynomial.legendre import leggauss
    xg, _ = leggauss(n)
    nx, ny = tm.shape
    for sx in range(nx):
        U = 0.5 * (tm.u_bounds[sx] + tm.u_bounds[sx + 1]) + 0.5 * (
            tm.u_bounds[sx + 1] - tm.u_bounds[sx]) * xg
        for sy in range(ny):
            Vv = 0.5 * (tm.v_bounds[sy] + tm.v_bounds[sy + 1]) + 0.5 * (
                tm.v_bounds[sy + 1] - tm.v_bounds[sy]) * xg
            _X, _Y, xu, xv, yu, yv = tm.geom(sx, sy, U, Vv)
            det = xu * yv - xv * yu
            if np.all(np.isfinite(det)) and float(np.min(det)) > 0.0:
                continue
            who = set(gu.owners[sx]) | set(gu.owners[sx + 1]) | set(
                gv.owners[sy]) | set(gv.owners[sy + 1])
            for key in ((sx, sy), (sx + 1, sy), (sx, sy + 1),
                        (sx + 1, sy + 1)):
                who |= {w for _xy, w in vclaims.get(key, ())}
            for ekey in (("h", sx, sy), ("h", sx, sy + 1), ("v", sx, sy),
                         ("v", sx + 1, sy)):
                who |= {w for _c, w in eclaims.get(ekey, ())}
            who.discard("the unit-cell edge")
            raise ValueError(
                f"compile_shapes: the merged map FOLDS (det J <= 0, min "
                f"{float(np.nanmin(det)):.3e}) in the cell u in "
                f"[{_fmt(tm.u_bounds[sx])}, {_fmt(tm.u_bounds[sx + 1])}], "
                f"v in [{_fmt(tm.v_bounds[sy])}, "
                f"{_fmt(tm.v_bounds[sy + 1])}], which is bounded by "
                f"{', '.join(sorted(who)) or 'the cell edges'}.  An outline "
                f"bulges past a neighbouring wall: two outlines are too close "
                f"in plan view, or a straight wall of one shape crosses the "
                f"arc bulge of another shape (even far away along that wall "
                f"-- e.g. pillars of different radii in one row).  Move them "
                f"apart (or align the straight wall with the curved shape's "
                f"bounding box).  If the shapes are in DIFFERENT layers, "
                f"layer_grids='per-layer' gives each layer its own map "
                f"(Phase E2); shapes in ONE layer have no route yet (the "
                f"hybrid merge is deferred).")


def compile_shapes(period_x, period_y, shapes, background_eps, *,
                   grid_hint=None, background_mu=None, with_mu=False):
    """Lay out a list of shape primitives as the staggered solver's wall grid
    and coordinate map: ``(eps_cell, x_walls, y_walls, cmap)``.

    What it does, in physical terms.  You describe ONE layer's cross-section
    by its physical outlines (:class:`Rect`, :class:`FilletRect`,
    :class:`Circle`, :class:`Ellipse`, :class:`SinusoidalWall`), painted in
    order onto ``background_eps``.  This function places the solver's walls
    so that every outline is a grid line -- every corner, 45-degree point and
    tangency point a grid vertex at its own physical position, every arc an
    exact grid edge -- squares the grid up (``Nx == Ny``) by halving the
    widest segments of the shorter axis, validates the map (no fold, lattice
    periodicity, outlines that do not cross), and reads off the permittivity
    of every ``(u, v)`` cell.

    Parameters
    ----------
    period_x, period_y : float
        The unit-cell periods (metres).
    shapes : sequence of shape primitives
        Painted in order: a later shape covers an earlier one.
    background_eps : complex or (3, 3)
        The permittivity where no shape is painted (PUBLIC ``Im > 0`` loss).
    grid_hint : int or (int, int), optional
        The minimum number of segments per axis; the widest segments are
        halved until it is met (and the grid stays square).  Use it to add
        resolution where the map is steep without moving any outline.
    background_mu : complex or (3, 3), optional
        The relative permeability where no shape is painted (default 1).
    with_mu : bool, optional
        Return the permeability grid as a FIFTH output (``None`` when no
        material is magnetic).  Required as soon as any shape carries ``mu=``
        or ``background_mu`` is given -- the four-output form would drop the
        permeability silently, so it RAISES instead.

    Returns
    -------
    eps_cell : ``(N, N)`` complex array (``(N, N, 3, 3)`` if any material is
        a tensor) -- the permittivity of each ``(u, v)`` cell.  A tensor is
        BLOCK-FORM (in-plane anisotropy: ``e13 = e23 = e31 = e32 = 0``) and
        is routed under a curved map since Phase D (the congruence
        ``sqrt(g) J^-1 eps J^-T`` at every quadrature node); an out-of-plane
        tensor under a curved map raises in the solver (Phase E).
    x_walls, y_walls : ``(N + 1,)`` float arrays -- the ``(u, v)`` wall grid
        (boundary arrays from 0 to the period).  Where the map is the
        identity -- on the cell edges and away from every curve -- these are
        the physical wall positions.
    cmap : the coordinate map
        (:class:`~lumenairy.elements.pmm._curvemap.TransfiniteMap`, or a
        :class:`~lumenairy.elements.pmm._curvemap.RefinedMap` of one when the
        grid was squared up), fingerprinted.  For rectangles only it is the
        identity map.
    mu_cell : (only with ``with_mu=True``) the ``(N, N)`` /
        ``(N, N, 3, 3)`` permeability of each cell, or ``None``.

    The four outputs feed the explicit-map entry
    ``pmm_jones_2d_staggered(px, py, eps_cell, ..., cmap=cmap)`` or
    ``PMM2DStackPure(..., cmap=cmap).add_layer(t, eps_cell=eps_cell)`` -- the
    same bytes as the ``shapes=`` entries (which call this machinery).  The
    ``shapes=`` entries are the everyday route; they also run a
    rectangles-only geometry on the shipped unmapped solver.

    Raises ``ValueError`` (naming the shapes) for crossing outlines, two
    different curves on one cell edge, a folding map, a sliver segment
    (below 1e-3 of the period), a fillet radius below sqrt(2) x 1e-3 of
    the period,
    or a shape outside the unit cell.

    Example -- a circular hole (r = 300 nm) in a silicon slab::

        from lumenairy.elements.pmm import Circle, Rect, compile_shapes
        slab = Rect(0.6e-6, 0.6e-6, 1.2e-6, 1.2e-6, eps=12.1)
        hole = Circle(0.6e-6, 0.6e-6, 0.3e-6, eps=1.0)
        eps_cell, xw, yw, cmap = compile_shapes(1.2e-6, 1.2e-6,
                                                [slab, hole], 1.0)
        # eps_cell == [[12.1, 12.1, 12.1], [12.1, 1, 12.1], [12.1, 12.1, 12.1]]
    """
    shapes = list(shapes)
    if not shapes:
        raise ValueError("compile_shapes: pass at least one shape.")
    U, Vb, cmap, cells, _ident, mus = _merge(
        period_x, period_y, [(None, shapes, background_eps, background_mu)],
        grid_hint)
    if with_mu:
        return cells[0], U, Vb, cmap, mus[0]
    if mus[0] is not None:
        raise ValueError(
            "compile_shapes: a shape (or background_mu) carries a "
            "permeability mu, which the four-output form would drop; pass "
            "with_mu=True to receive the mu_cell as a fifth output.")
    return cells[0], U, Vb, cmap
