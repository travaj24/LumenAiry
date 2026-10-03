"""Coordinate maps for curved in-plane cells in the pure staggered 2-D PMM.
==========================================================================

What this module is for
-----------------------
The pure staggered 2-D PMM (:func:`~lumenairy.elements.pmm.pmm_jones_2d_staggered`,
:class:`~lumenairy.elements.pmm.PMM2DStackPure`) divides the unit cell into a
grid of rectangles -- the **wall grid** -- and expands the field in each
rectangle in polynomials.  Without help, every material boundary must be one
of the grid's straight walls.

A **coordinate map** ``(x, y) = Phi(u, v)`` is a smooth distortion of the
plane.  The solver keeps working on a straight rectangular grid in the new
coordinates ``(u, v)``; the map bends (or stretches) that grid in the real
``(x, y)`` plane.  The map does not depend on the height ``z``, and ONE map is
shared by every layer of a stack and by both half-spaces (the map is owned by
the STACK).

The words used below:

* ``u``, ``v`` -- the solver's coordinates (metres, same period as ``x``,
  ``y``); ``(sx, sy)`` indexes a rectangle (a **cell**) of the ``(u, v)`` wall
  grid.
* **The Jacobian** ``J = d(x, y)/d(u, v) = [[x_u, x_v], [y_u, y_v]]`` -- the
  2x2 matrix of the map's local stretch and shear (``x_u = dx/du`` and so on).
  ``det J`` is the local AREA magnification; it must be positive (the map
  must not fold the plane over itself).
* **The metric** ``g = J^T J`` -- the local change of lengths and angles;
  ``sqrt(g)`` below means ``det J``.
* **Covariant components** ``E' = J^T E`` -- the field components measured
  ALONG the bent grid lines (``E'_u = E . dPhi/du``).  In ``(u, v)``
  coordinates Maxwell's equations keep their Cartesian form if every material
  is replaced by the **effective tensors** ``eps' = det J J^-1 eps J^-T`` and
  ``mu' = det J J^-1 mu J^-T`` (transformation optics; Weiss et al.,
  Opt. Express 17, 8051 (2009); Essig & Busch, Opt. Express 18, 23258 (2010)).
  Even vacuum becomes an anisotropic magnetic medium under a map, which is why
  every mapped region runs the solver's magnetic + tensor route.

The map protocol
----------------
A map is any object with

* ``period_x``, ``period_y`` -- the lattice periods;
* ``u_walls``, ``v_walls`` -- the ``(u, v)`` wall grid, each either an ``int``
  ``N`` (the uniform lattice, exactly as the solver's own ``wx`` / ``wy``) or
  an increasing ``(N + 1,)`` boundary array running ``0 .. period``; and the
  resolved boundary arrays ``u_bounds``, ``v_bounds``;
* ``geom(sx, sy, U, V) -> (X, Y, x_u, x_v, y_u, y_v)`` -- given a cell index
  and 1-D arrays ``U`` (length ``nu``) and ``V`` (length ``nv``) of ``u`` and
  ``v`` values INSIDE that cell, the physical position and the four Jacobian
  entries on the ``(nu, nv)`` tensor grid of those points (first index along
  ``u``).  The solver calls it on Gauss-Legendre nodes, so it never asks for a
  value on a cell edge -- a map may be only continuous (not differentiable)
  across a grid line, and the per-cell evaluation is what lets the Jacobian
  jump there;
* ``fingerprint`` -- a content string (class, walls, periods, curve
  parameters) that is equal for two maps exactly when they describe the same
  geometry, used to key caches.

:class:`CellMap` is the base class that supplies the shared validation;
:class:`IdentityMap` and :class:`SeparableStretch` (Phase A) bend nothing;
:class:`TransfiniteMap` (Phase B) is the general CURVED map: you name grid
edges that must be curves -- :class:`Line`, :class:`Arc`,
:class:`EllipseArc`, :class:`Sinusoid` -- and it fills in every cell by
blending that cell's four edge curves, so a circle, an ellipse, a rounded
corner or a sinusoidal wall becomes an exact grid line.  It is still an
expert-level object (you lay out the walls and the curves yourself, as the
private ``_circle_map_3x3`` / ``_circle_map_5x5`` / ``_fillet_map_5x5`` /
``_ellipse_map_3x3`` / ``_sine_stripe_map_3x3`` builders below do); the shape
primitives that do it for you are Phase C.

The singular vertices.  A smooth closed curve made of grid lines of a tensor
grid must turn 90 degrees in ``(u, v)`` where it does not turn at all in
``(x, y)`` -- four such cell corners per curve, where ``det J = 0`` and the
effective tensors grow like ``1 / distance``.  They are allowed (at cell
corners only, listed by ``singular_vertices``), and the solver integrates
the cells that own one with a Duffy-collapsed rule
(:func:`~lumenairy.elements.pmm.twod_staggered._stag_duffy_points`).

Validation (``CellMap.validate``)
---------------------------------
* **Lattice periodicity.**  ``Phi(u + p_x, v) = Phi(u, v) + (p_x, 0)`` and
  ``Phi(u, v + p_y) = Phi(u, v) + (0, p_y)``, and the derivative ALONG each
  side of the cell is periodic (``d Phi / dv`` on the sides ``u = 0, p_x``,
  ``d Phi / du`` on ``v = 0, p_y``).  This is what keeps the solver's Bloch
  boundary condition (the phase ``tau`` that glues the cell's two edges)
  unchanged under the map -- the unknown that must be continuous across the
  side ``u = 0 ~ p_x`` is ``E'_v = E . dPhi/dv`` -- and what makes the image
  of the ``(u, v)`` cell one full period of the physical lattice.  The
  derivative ACROSS a side (``d Phi / du`` on ``u = 0, p_x``) need NOT be
  periodic, exactly as it may jump across every interior grid line: a
  transfinite map is only ``C0`` there, and its covariant ``E'_u`` is a
  broken (discontinuous) unknown across ``u = const`` lines (Phase B build
  finding: requiring the full Jacobian periodic refused every curved map,
  e.g. the 3 x 3 circle by 6.2e-2).
* **The cell boundary maps onto itself**: the side ``u = 0`` lands on the line
  ``x = 0`` and the side ``v = 0`` on ``y = 0``, so the map frame and the lab
  share one lattice origin.  Pointwise IDENTITY on the boundary (which the
  curved shape maps will have) is stronger than the solver needs -- a
  separable stretch slides points ALONG the boundary and is still exact (the
  build's gates A3 / A4 measure exactly that case).
* ``det J > 0`` -- checked here on a probe grid, and again by the solver at
  EVERY interior quadrature node it actually uses (a map may be singular at
  isolated cell CORNERS, which no Gauss node touches).
"""
from __future__ import annotations

import hashlib

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["Arc", "CellMap", "EdgeCurve", "EllipseArc", "IdentityMap", "Line",
           "RefinedMap", "SeparableStretch", "SineStretch", "Sinusoid",
           "TransfiniteMap"]

#: Relative tolerance of the map validation (periodicity and the boundary
#: check), in units of the period for positions and of the Jacobian scale for
#: derivatives.  The maps shipped here are analytic, so a correct map meets
#: these at round-off (~1e-16); 1e-12 leaves four decades for the arithmetic
#: of user-supplied curves and still refuses any real (>= 1e-9 p) defect.
_MAP_VALIDATE_TOL = 1e-12


def _resolve_walls(walls, period, axis):
    """``(spec, bounds)``: the wall spec as given (``int`` or array) and the
    resolved ``(N + 1,)`` boundary array.  An ``int`` resolves through
    ``np.linspace`` exactly as :class:`~lumenairy.elements.pmm.twod_staggered.Basis1D`
    does on its uniform path, so the two agree to the bit."""
    if np.ndim(walls) == 0:
        n = int(walls)
        if n < 1:
            raise ValueError(f"{axis}: the segment count must be >= 1, got "
                             f"{walls!r}.")
        if period is None:
            raise ValueError(f"{axis}: an integer wall count needs the "
                             f"period (period_x / period_y).")
        return n, np.linspace(0.0, float(period), n + 1)
    b = np.asarray(walls, dtype=float).ravel()
    if b.size < 2 or np.any(np.diff(b) <= 0.0):
        raise ValueError(f"{axis}: walls must be an int N or a STRICTLY "
                         f"increasing (N + 1,) boundary array, got {walls!r}.")
    if abs(b[0]) > 1e-13 * abs(b[-1]):
        raise ValueError(f"{axis}: the boundary array must start at 0, got "
                         f"{b[0]!r}.")
    if period is not None and abs(b[-1] - float(period)) > 1e-13 * float(period):
        raise ValueError(f"{axis}: the boundary array must end at the period "
                         f"{period!r}, got {b[-1]!r}.")
    b = b.copy()
    b[0] = 0.0
    return b, b


class CellMap:
    """Base class of the ``(u, v) -> (x, y)`` map protocol (see the module
    docstring for every term).  Subclasses set ``period_x``, ``period_y``,
    ``u_walls``, ``v_walls``, ``u_bounds``, ``v_bounds`` and implement
    :meth:`geom` and :meth:`_key`; the constructor of a subclass ends with
    ``self.validate()``."""

    period_x: float
    period_y: float
    u_bounds: np.ndarray
    v_bounds: np.ndarray

    def _init_walls(self, u_walls, v_walls, period_x, period_y):
        self.u_walls, self.u_bounds = _resolve_walls(u_walls, period_x,
                                                     "u_walls")
        self.v_walls, self.v_bounds = _resolve_walls(v_walls, period_y,
                                                     "v_walls")
        self.period_x = float(self.u_bounds[-1])
        self.period_y = float(self.v_bounds[-1])

    @property
    def shape(self):
        """``(Nx, Ny)`` -- the cell count of the ``(u, v)`` wall grid."""
        return (int(self.u_bounds.size - 1), int(self.v_bounds.size - 1))

    def geom(self, sx, sy, U, V):  # pragma: no cover - protocol
        """``(X, Y, x_u, x_v, y_u, y_v)`` on the ``(len(U), len(V))`` tensor
        grid of points ``U`` x ``V`` inside cell ``(sx, sy)``."""
        raise NotImplementedError

    def _key(self):  # pragma: no cover - protocol
        raise NotImplementedError

    def geom_points(self, sx, sy, U, V):
        """``(X, Y, x_u, x_v, y_u, y_v)`` at the POINTS ``(U[k], V[k])``
        (equal-length 1-D arrays) inside cell ``(sx, sy)`` -- the pointwise
        companion of :meth:`geom`, used by the solver's corner (Duffy) rule,
        whose nodes are not a tensor grid.  This default evaluates through
        :meth:`geom`, one call per distinct ``U`` (or ``V``, whichever is
        fewer); a map with a vectorised pointwise form overrides it."""
        U = np.asarray(U, dtype=float).ravel()
        V = np.asarray(V, dtype=float).ravel()
        out = [np.empty(U.size) for _ in range(6)]
        by_u = np.unique(U).size <= np.unique(V).size
        key, other = (U, V) if by_u else (V, U)
        uniq, inv = np.unique(key, return_inverse=True)
        for k, val in enumerate(uniq):
            sel = np.nonzero(inv == k)[0]
            if by_u:
                g = self.geom(sx, sy, np.array([val]), other[sel])
                rows = [a[0, :] for a in g]
            else:
                g = self.geom(sx, sy, other[sel], np.array([val]))
                rows = [a[:, 0] for a in g]
            for o, r in zip(out, rows):
                o[sel] = r
        return tuple(out)

    @property
    def singular_vertices(self):
        """Cell corners where ``det J = 0`` (see
        :attr:`TransfiniteMap.singular_vertices`); none for a map whose
        Jacobian is regular everywhere, which is this default."""
        return []

    @property
    def fingerprint(self):
        """Content fingerprint: equal for two maps exactly when they describe
        the same geometry (class, wall grid, periods and curve parameters).
        A hex SHA-256 string, so it can key any cache."""
        h = hashlib.sha256()
        h.update(type(self).__name__.encode())
        h.update(np.asarray([self.period_x, self.period_y]).tobytes())
        h.update(np.ascontiguousarray(self.u_bounds).tobytes())
        h.update(b"|")
        h.update(np.ascontiguousarray(self.v_bounds).tobytes())
        h.update(repr(self._key()).encode())
        return h.hexdigest()

    # ------------------------------------------------------------ validation
    def _nodes(self, s, bounds, n):
        xg, _ = leggauss(n)
        return 0.5 * (bounds[s] + bounds[s + 1]) + 0.5 * (
            bounds[s + 1] - bounds[s]) * xg

    def validate(self, n=7):
        """Raise ``ValueError`` unless the map is lattice-periodic, maps the
        cell boundary onto itself, and has ``det J > 0`` on an ``n x n``
        Gauss probe grid of every cell.  Run by every shipped constructor;
        the solver re-checks ``det J`` at its own quadrature nodes."""
        px, py = self.period_x, self.period_y
        Nx, Ny = self.shape
        tol = _MAP_VALIDATE_TOL
        for sx in range(Nx):
            U = self._nodes(sx, self.u_bounds, n)
            for sy in range(Ny):
                V = self._nodes(sy, self.v_bounds, n)
                X, Y, xu, xv, yu, yv = self.geom(sx, sy, U, V)
                det = xu * yv - xv * yu
                if not np.all(np.isfinite(det)) or float(np.min(det)) <= 0.0:
                    raise ValueError(
                        f"{type(self).__name__}: det J must be > 0 inside "
                        f"every cell (the map must not fold the plane); cell "
                        f"({sx}, {sy}) reads min det J = "
                        f"{float(np.nanmin(det)):.3e}.")
        # periodicity in u: the image of u = p_x is the image of u = 0
        # shifted by one period, with an equal Jacobian; and the side u = 0
        # lands on x = 0 (the boundary maps onto itself)
        for sy in range(Ny):
            V = self._nodes(sy, self.v_bounds, n)
            a = self.geom(0, sy, np.array([self.u_bounds[0]]), V)
            b = self.geom(Nx - 1, sy, np.array([self.u_bounds[-1]]), V)
            self._check_periodic(a, b, (px, 0.0), "u", sy, tol)
            if float(np.max(np.abs(a[0]))) > tol * px:
                raise ValueError(
                    f"{type(self).__name__}: the cell side u = 0 must map onto "
                    f"the line x = 0 (row {sy}: max |x| = "
                    f"{float(np.max(np.abs(a[0]))):.3e}).")
        for sx in range(Nx):
            U = self._nodes(sx, self.u_bounds, n)
            a = self.geom(sx, 0, U, np.array([self.v_bounds[0]]))
            b = self.geom(sx, Ny - 1, U, np.array([self.v_bounds[-1]]))
            self._check_periodic(a, b, (0.0, py), "v", sx, tol)
            if float(np.max(np.abs(a[1]))) > tol * py:
                raise ValueError(
                    f"{type(self).__name__}: the cell side v = 0 must map onto "
                    f"the line y = 0 (column {sx}: max |y| = "
                    f"{float(np.max(np.abs(a[1]))):.3e}).")
        return self

    def _check_periodic(self, a, b, shift, axis, idx, tol):
        X0, Y0 = a[0], a[1]
        X1, Y1 = b[0], b[1]
        scale = max(self.period_x, self.period_y)
        dpos = max(float(np.max(np.abs(X1 - X0 - shift[0]))),
                   float(np.max(np.abs(Y1 - Y0 - shift[1]))))
        # only the TANGENTIAL derivative of the side must be periodic: on
        # the u-sides (u = 0, p_x) that is d/dv = (x_v, y_v), on the v-sides
        # d/du = (x_u, y_u); the normal derivative may jump (the map is C0
        # across every grid line, the periodic seam included)
        tang = (3, 5) if axis == "u" else (2, 4)
        jscale = max(1.0, max(float(np.max(np.abs(a[k]))) for k in tang))
        djac = max(float(np.max(np.abs(b[k] - a[k]))) for k in tang)
        if dpos > tol * scale or djac > tol * jscale:
            raise ValueError(
                f"{type(self).__name__}: the map must be lattice-periodic -- "
                f"Phi(u + p_x, v) = Phi(u, v) + (p_x, 0) and Phi(u, v + p_y) "
                f"= Phi(u, v) + (0, p_y) with a periodic derivative along "
                f"each side of the cell -- so that "
                f"the solver's Bloch glue is unchanged; along {axis} (index "
                f"{idx}) the position mismatch is {dpos:.3e} and the "
                f"Jacobian mismatch {djac:.3e}.")


class IdentityMap(CellMap):
    """The identity map ``(x, y) = (u, v)`` on a given wall grid.

    Physically a no-op.  It exists as the reference arm of the variable-
    coefficient (quadrature) assembly: with ``cmap=IdentityMap(...)`` the
    solver runs the mapped code path end to end -- the magnetic + tensor
    route, every weighted block by 2-D Gauss quadrature, the cofactor far
    field -- and must reproduce the unmapped solver to round-off (gate A2).
    It is NOT the same bytes as ``cmap=None`` (summation order differs), which
    is why "no map" is a dispatch to the shipped code and not this object."""

    def __init__(self, u_walls, v_walls, period_x=None, period_y=None):
        self._init_walls(u_walls, v_walls, period_x, period_y)
        self.validate()

    def geom(self, sx, sy, U, V):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        X = np.broadcast_to(U[:, None], (U.size, V.size)).astype(float)
        Y = np.broadcast_to(V[None, :], (U.size, V.size)).astype(float)
        one = np.ones_like(X)
        zero = np.zeros_like(X)
        return X, Y, one, zero, zero, one

    def _key(self):
        return ()


class SineStretch:
    """A periodic 1-D stretch of one axis: ``x = f(u) = u + a sin(2 pi u / p)``
    with ``f'(u) = 1 + (2 pi a / p) cos(2 pi u / p)``.

    ``amplitude`` is ``a`` in metres; ``p`` is the period of the axis the
    stretch is attached to (supplied by :class:`SeparableStretch`).  The map is
    one-to-one exactly when ``|a| < p / (2 pi)`` (then ``f' > 0``
    everywhere); it COMPRESSES the grid where ``f' < 1`` and spreads it where
    ``f' > 1`` -- at ``a = 0.15 p`` the local stretch runs from 0.058 to 1.94,
    a 33:1 range.  ``f(u + p) = f(u) + p``, so the stretch is
    lattice-periodic, and ``f(0) = 0``.

    This is Granet's "adaptive spatial resolution" idea for the polynomial
    basis: concentrate resolution where the field needs it without moving the
    physical walls."""

    def __init__(self, amplitude):
        self.amplitude = float(amplitude)

    def __call__(self, u, period):
        """``(f(u), f'(u))`` for the array ``u``."""
        k = 2.0 * np.pi / float(period)
        u = np.asarray(u, dtype=float)
        return (u + self.amplitude * np.sin(k * u),
                1.0 + self.amplitude * k * np.cos(k * u))

    def check(self, period, axis):
        """Refuse a non-invertible stretch (``|a| >= p / (2 pi)``)."""
        if not abs(self.amplitude) * 2.0 * np.pi / float(period) < 1.0:
            raise ValueError(
                f"SineStretch on {axis}: |amplitude| = {abs(self.amplitude)!r} "
                f"must be < period / (2 pi) = {float(period) / (2 * np.pi)!r} "
                f"-- beyond it f'(u) changes sign and the map folds the "
                f"plane (det J <= 0).")

    def inverse(self, x, period):
        """``u = f^-1(x)`` (safeguarded Newton on the monotone ``f``)."""
        x = np.asarray(x, dtype=float)
        u = x.copy()
        for _ in range(100):
            f, fp = self(u, period)
            du = (f - x) / fp
            u = u - du
            if float(np.max(np.abs(du), initial=0.0)) <= 1e-16 * float(period):
                break
        return u

    def key(self):
        return ("SineStretch", self.amplitude)


class SeparableStretch(CellMap):
    """A separable per-axis stretch ``x = f(u)``, ``y = h(v)`` on a given
    ``(u, v)`` wall grid -- the map of Phase A of the curved-cell build.

    ``fx`` / ``fy`` are 1-D stretches (:class:`SineStretch`), ``None`` meaning
    the identity on that axis.  The Jacobian is diagonal, ``J = diag(f'(u),
    h'(v))``, with analytic derivatives, so ``det J = f' h'`` and the metric
    has no shear (``g12 = 0``).

    A separable stretch does not curve any wall: it moves the physical image
    of each ``(u, v)`` wall to ``f(u_wall)``.  To model a GIVEN physical
    structure, place the ``(u, v)`` walls at the PREIMAGES of the physical
    walls -- :meth:`from_physical_walls` does exactly that.  The structure is
    then unchanged and only the solver's resolution is redistributed, so the
    mapped solve must converge to the SAME answer as the unmapped one: that is
    the self-consistency gate of this phase."""

    def __init__(self, u_walls, v_walls, fx=None, fy=None, period_x=None,
                 period_y=None):
        self._init_walls(u_walls, v_walls, period_x, period_y)
        self.fx = fx
        self.fy = fy
        if fx is not None:
            fx.check(self.period_x, "x")
        if fy is not None:
            fy.check(self.period_y, "y")
        self.validate()

    @classmethod
    def from_physical_walls(cls, x_walls, y_walls, fx=None, fy=None):
        """The stretch whose ``(u, v)`` walls are the PREIMAGES of the given
        physical boundary arrays (each an increasing ``(N + 1,)`` array from 0
        to the period), so the physical structure is exactly the one those
        walls describe."""
        xb = np.asarray(x_walls, dtype=float).ravel()
        yb = np.asarray(y_walls, dtype=float).ravel()
        px, py = float(xb[-1]), float(yb[-1])
        ub = xb.copy() if fx is None else fx.inverse(xb, px)
        vb = yb.copy() if fy is None else fy.inverse(yb, py)
        ub[0], ub[-1] = 0.0, px
        vb[0], vb[-1] = 0.0, py
        return cls(ub, vb, fx=fx, fy=fy)

    def physical_walls(self):
        """``(x_bounds, y_bounds)`` -- the physical images of the wall grid."""
        fx = (self.u_bounds if self.fx is None
              else self.fx(self.u_bounds, self.period_x)[0])
        fy = (self.v_bounds if self.fy is None
              else self.fy(self.v_bounds, self.period_y)[0])
        return fx, fy

    def geom(self, sx, sy, U, V):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if self.fx is None:
            fu, dfu = U, np.ones_like(U)
        else:
            fu, dfu = self.fx(U, self.period_x)
        if self.fy is None:
            hv, dhv = V, np.ones_like(V)
        else:
            hv, dhv = self.fy(V, self.period_y)
        shp = (U.size, V.size)
        X = np.broadcast_to(fu[:, None], shp).astype(float)
        Y = np.broadcast_to(hv[None, :], shp).astype(float)
        xu = np.broadcast_to(dfu[:, None], shp).astype(float)
        yv = np.broadcast_to(dhv[None, :], shp).astype(float)
        zero = np.zeros(shp)
        return X, Y, xu, zero, zero, yv

    def geom_points(self, sx, sy, U, V):
        """The pointwise form of :meth:`geom`, vectorised (the base-class
        default loops over distinct ``U``; the curved mortar of Phase E2
        evaluates a separable map at thousands of scattered nodes)."""
        U = np.asarray(U, dtype=float).ravel()
        V = np.asarray(V, dtype=float).ravel()
        if self.fx is None:
            fu, dfu = U.copy(), np.ones_like(U)
        else:
            fu, dfu = self.fx(U, self.period_x)
        if self.fy is None:
            hv, dhv = V.copy(), np.ones_like(V)
        else:
            hv, dhv = self.fy(V, self.period_y)
        zero = np.zeros(U.shape)
        return (np.asarray(fu, dtype=float), np.asarray(hv, dtype=float),
                np.asarray(dfu, dtype=float), zero, zero.copy(),
                np.asarray(dhv, dtype=float))

    def _key(self):
        return (None if self.fx is None else self.fx.key(),
                None if self.fy is None else self.fy.key())


# =========================================================================== #
# Phase B: edge curves and the Gordon-Hall transfinite map
# =========================================================================== #
#: Endpoint tolerance of an edge curve against its two vertex images, in
#: units of the larger period.  The shipped curves are analytic, so a correct
#: curve meets it at round-off (~1e-16); 1e-12 refuses any genuine mismatch
#: (a vertex moved by >= 1e-12 p would open a crack between two cells).
_EDGE_END_TOL = 1e-12


class EdgeCurve:
    """One curved EDGE of the ``(u, v)`` wall grid, parametrised on
    ``s in [0, 1]``.

    Calling the curve on an array ``s`` returns ``(val, der)``: the physical
    points ``(len(s), 2)`` and the derivative ``d(x, y)/ds`` ``(len(s), 2)``
    (analytic, not differenced).  ``s = 0`` is the edge's LOWER-index vertex
    (the vertex with the smaller ``u`` for a horizontal edge, the smaller
    ``v`` for a vertical one) and ``s = 1`` the other.  The parametrisation
    matters: the transfinite map places the ``(u, v)`` grid line's points
    along the curve at the curve's own ``s``, so a curve parametrised
    uniformly in angle (an arc) or in the running coordinate (a sinusoid)
    keeps the map smooth inside the cells on both sides."""

    def __call__(self, s):  # pragma: no cover - protocol
        raise NotImplementedError

    def key(self):  # pragma: no cover - protocol
        """Content tuple of every parameter (enters the map fingerprint)."""
        raise NotImplementedError

    def endpoints(self):
        """``(start, end)`` -- the physical images of ``s = 0`` and ``1``."""
        v, _ = self(np.array([0.0, 1.0]))
        return v[0], v[1]

    def piece(self, s0, s1):  # pragma: no cover - protocol
        """The sub-curve from ``s = s0`` to ``s = s1``, re-parametrised on
        ``[0, 1]`` and UNIFORM in the same parameter (Phase C: a wall of
        another shape that crosses this edge splits it into pieces, and the
        transfinite blend of each sub-cell must place its points exactly where
        the whole edge did).  ``piece(0, 1)`` is the curve itself."""
        raise NotImplementedError


class Line(EdgeCurve):
    """The straight segment from ``P`` (``s = 0``) to ``Q`` (``s = 1``),
    uniform in ``s``.  Every edge of a :class:`TransfiniteMap` that is not
    given a curve is this line between its two vertex images."""

    def __init__(self, P, Q):
        self.P = np.asarray(P, dtype=float).reshape(2)
        self.Q = np.asarray(Q, dtype=float).reshape(2)

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        d = self.Q - self.P
        return (self.P[None, :] + s[:, None] * d[None, :],
                np.broadcast_to(d[None, :], (s.size, 2)).copy())

    def key(self):
        return ("Line", tuple(self.P), tuple(self.Q))

    def piece(self, s0, s1):
        if s0 == 0.0 and s1 == 1.0:
            return self
        d = self.Q - self.P
        return Line(self.P + float(s0) * d, self.P + float(s1) * d)


class Arc(EdgeCurve):
    """A circular arc of ``radius`` about ``center``, from polar angle
    ``theta0`` (``s = 0``) to ``theta1`` (``s = 1``) in RADIANS, uniform in
    angle: ``(x, y) = c + r (cos th, sin th)``, ``th = th0 + s (th1 - th0)``.
    ``theta1 < theta0`` runs clockwise.  :meth:`through` builds the arc that
    joins two given grid-vertex images about a given centre."""

    def __init__(self, center, radius, theta0, theta1):
        self.center = np.asarray(center, dtype=float).reshape(2)
        self.radius = float(radius)
        self.theta0 = float(theta0)
        self.theta1 = float(theta1)
        if not self.radius > 0.0:
            raise ValueError(f"Arc: radius must be > 0, got {radius!r}.")
        if self.theta0 == self.theta1:
            raise ValueError("Arc: theta0 == theta1 is a point, not an edge.")

    @classmethod
    def through(cls, P, Q, center, radius=None):
        """The SHORTER arc about ``center`` from the point ``P`` to the point
        ``Q`` (its sweep is below 180 degrees).  Both points must sit on one
        circle about ``center`` (and on ``radius``, if given) to
        :data:`_EDGE_END_TOL` relative -- otherwise there is no such arc."""
        P = np.asarray(P, dtype=float).reshape(2)
        Q = np.asarray(Q, dtype=float).reshape(2)
        c = np.asarray(center, dtype=float).reshape(2)
        rP = float(np.hypot(*(P - c)))
        rQ = float(np.hypot(*(Q - c)))
        r = rP if radius is None else float(radius)
        tol = _EDGE_END_TOL * max(r, 1.0)
        if abs(rP - r) > tol or abs(rQ - r) > tol:
            raise ValueError(
                f"Arc.through: the two points are not on one circle of "
                f"radius {r!r} about {tuple(c)} (distances {rP!r}, {rQ!r}).")
        t0 = float(np.arctan2(P[1] - c[1], P[0] - c[0]))
        t1 = float(np.arctan2(Q[1] - c[1], Q[0] - c[0]))
        dt = (t1 - t0 + np.pi) % (2.0 * np.pi) - np.pi
        if abs(abs(dt) - np.pi) < 1e-12:
            raise ValueError("Arc.through: the points are diametrically "
                             "opposite -- the shorter arc is ambiguous; give "
                             "the angles explicitly.")
        return cls(c, r, t0, t0 + dt)

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        sw = self.theta1 - self.theta0
        th = self.theta0 + s * sw
        cs, sn = np.cos(th), np.sin(th)
        val = self.center[None, :] + self.radius * np.stack([cs, sn], 1)
        der = (self.radius * sw) * np.stack([-sn, cs], 1)
        return val, der

    def key(self):
        return ("Arc", tuple(self.center), self.radius, self.theta0,
                self.theta1)

    def piece(self, s0, s1):
        if s0 == 0.0 and s1 == 1.0:
            return self
        sw = self.theta1 - self.theta0
        return Arc(self.center, self.radius, self.theta0 + float(s0) * sw,
                   self.theta0 + float(s1) * sw)


class EllipseArc(EdgeCurve):
    """An arc of the ellipse with ``semi_axes = (a, b)`` about ``center``,
    rotated by ``angle`` (radians, counter-clockwise), from the PARAMETRIC
    angle ``t0`` (``s = 0``) to ``t1`` (``s = 1``), uniform in ``t``:
    ``(x, y) = c + Rot(angle) (a cos t, b sin t)``.  ``a = b`` is a circular
    :class:`Arc`.  (The parametric angle is not the polar angle unless
    ``a = b``; the point at ``t = 45`` degrees is ``(a, b) / sqrt 2``.)"""

    def __init__(self, center, semi_axes, t0, t1, angle=0.0):
        self.center = np.asarray(center, dtype=float).reshape(2)
        a, b = (float(v) for v in semi_axes)
        if not (a > 0.0 and b > 0.0):
            raise ValueError(f"EllipseArc: semi-axes must be > 0, got "
                             f"{semi_axes!r}.")
        self.a, self.b = a, b
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.angle = float(angle)
        if self.t0 == self.t1:
            raise ValueError("EllipseArc: t0 == t1 is a point, not an edge.")

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        sw = self.t1 - self.t0
        t = self.t0 + s * sw
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        px, py = self.a * np.cos(t), self.b * np.sin(t)
        dx, dy = -self.a * np.sin(t) * sw, self.b * np.cos(t) * sw
        val = self.center[None, :] + np.stack([ca * px - sa * py,
                                               sa * px + ca * py], 1)
        der = np.stack([ca * dx - sa * dy, sa * dx + ca * dy], 1)
        return val, der

    def key(self):
        return ("EllipseArc", tuple(self.center), self.a, self.b, self.t0,
                self.t1, self.angle)

    def piece(self, s0, s1):
        if s0 == 0.0 and s1 == 1.0:
            return self
        sw = self.t1 - self.t0
        return EllipseArc(self.center, (self.a, self.b),
                          self.t0 + float(s0) * sw, self.t0 + float(s1) * sw,
                          angle=self.angle)


class Sinusoid(EdgeCurve):
    """A piece of the sinusoidal line ``x = base + A sin(2 pi y / period +
    phase)`` (``along = 'y'``: the line runs in ``y``) or ``y = base + A
    sin(2 pi x / period + phase)`` (``along = 'x'``), from the running
    coordinate ``t0`` (``s = 0``) to ``t1`` (``s = 1``), uniform in it.

    A full-period sinusoidal WALL is the chain of these pieces along one
    ``(u, v)`` grid line; with ``period`` equal to the lattice period along
    the line it is lattice-periodic."""

    def __init__(self, base, amplitude, period, t0, t1, phase=0.0,
                 along="y"):
        if along not in ("x", "y"):
            raise ValueError(f"Sinusoid: along must be 'x' or 'y', got "
                             f"{along!r}.")
        self.base = float(base)
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.phase = float(phase)
        self.along = along
        if not self.period > 0.0 or self.t0 == self.t1:
            raise ValueError("Sinusoid: period must be > 0 and t0 != t1.")

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        sw = self.t1 - self.t0
        t = self.t0 + s * sw
        k = 2.0 * np.pi / self.period
        w = self.base + self.amplitude * np.sin(k * t + self.phase)
        dw = self.amplitude * k * np.cos(k * t + self.phase) * sw
        dt = np.full_like(t, sw)
        if self.along == "y":
            return np.stack([w, t], 1), np.stack([dw, dt], 1)
        return np.stack([t, w], 1), np.stack([dt, dw], 1)

    def key(self):
        return ("Sinusoid", self.base, self.amplitude, self.period, self.t0,
                self.t1, self.phase, self.along)

    def piece(self, s0, s1):
        if s0 == 0.0 and s1 == 1.0:
            return self
        sw = self.t1 - self.t0
        return self.between(self.t0 + float(s0) * sw,
                            self.t0 + float(s1) * sw)

    def between(self, t0, t1):
        """The piece between the running coordinates ``t0`` and ``t1`` given
        EXACTLY (the parameter of a sinusoid is the running coordinate itself,
        so a piece cut at a wall position needs no ``s`` arithmetic)."""
        return Sinusoid(self.base, self.amplitude, self.period, t0, t1,
                        phase=self.phase, along=self.along)


class TransfiniteMap(CellMap):
    """The Gordon-Hall (bilinearly blended) TRANSFINITE map: the general
    curved-cell map of the pure staggered 2-D PMM.

    What it is, physically.  You describe the device's curved boundaries as
    EDGES of the solver's ``(u, v)`` wall grid: a circle of radius ``r``
    becomes, for instance, the four edges of the middle cell of a 3 x 3 grid,
    each a 90-degree :class:`Arc`.  Inside every rectangular ``(u, v)`` cell
    the map then fills in the interior by blending that cell's FOUR edge
    curves -- the classical construction of Gordon & Hall (Int. J. Numer.
    Meth. Eng. 7, 461 (1973)): with ``(s, t)`` the cell's own coordinates in
    ``[0, 1]^2`` and ``B, T, L, R`` its bottom, top, left and right edges,

        Phi(s, t) = (1 - t) B(s) + t T(s) + (1 - s) L(t) + s R(t)
                    - [(1 - s)(1 - t) P00 + s (1 - t) P10
                       + (1 - s) t P01 + s t P11]

    (``P..`` the four vertex images).  The map reproduces each edge curve
    exactly on its edge, so a material boundary drawn along those edges is
    represented EXACTLY (the mapped disk area equals ``pi r^2`` to
    round-off), and two neighbouring cells share their common edge curve, so
    the map is continuous across every grid line (``C0``); its derivative may
    jump there, which the solver's covariant (staggered) unknowns absorb.
    Where no edge is curved the blend of four straight edges is the bilinear
    map, and where in addition the vertex images are the grid vertices it is
    the IDENTITY.  Inside a cell the map is as smooth as its edge curves
    (analytic for lines, arcs, ellipse arcs and sinusoids).

    The singular vertices.  A smooth CLOSED curve made of grid lines of a
    tensor grid must turn by 90 degrees in ``(u, v)`` at (at least) four
    vertices where it does not turn at all in ``(x, y)`` -- the circle's four
    45-degree points.  There the cell's two edge tangents are antiparallel and
    ``det J = 0``; nearby the effective tensors grow like ``1 / distance``
    (integrable).  This is unavoidable on a tensor grid (the planning
    document, section 2.6) and is ALLOWED at a cell CORNER; a singular point
    INSIDE a cell is refused (by :meth:`validate` here and by the solver at
    every quadrature node).  :attr:`singular_vertices` lists them.

    Parameters
    ----------
    u_walls, v_walls : int or increasing ``(N + 1,)`` array
        The ``(u, v)`` wall grid, as for every map (``Nx == Ny`` is the
        solver's requirement, not the map's).
    vertex_images : ``(Nx + 1, Ny + 1, 2)`` array or ``None``
        Physical image of every grid vertex ``(u_i, v_j)``; ``None`` = the
        vertices themselves (identity at every vertex).
    curved_edges : mapping or ``None``
        ``{('h', i, j): curve, ('v', i, j): curve}``: ``('h', i, j)`` is the
        horizontal edge from vertex ``(i, j)`` to ``(i + 1, j)`` (along ``u``
        at ``v = v_j``), ``('v', i, j)`` the vertical edge from ``(i, j)`` to
        ``(i, j + 1)``.  Each curve's ``s = 0`` end must be the lower-index
        vertex's image and its ``s = 1`` end the other one (checked; a
        mismatch would tear the map open and RAISES).  Edges not listed are
        straight :class:`Line` s between their vertex images.
    period_x, period_y : float, only with integer walls.

    The constructor checks the edge endpoints, then :meth:`CellMap.validate`
    (lattice periodicity, the cell boundary onto itself, ``det J > 0`` inside
    every cell on a probe grid).  Plan:
    ``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md``, section 4.2; the
    construction is lifted from the planning probe's ``TransfiniteMap``
    (``validation/probe_pmm2d_curved/_curved_scratch.py``), whose circle and
    fillet maps landed on an independent 3-D finite-element oracle."""

    def __init__(self, u_walls, v_walls, vertex_images=None,
                 curved_edges=None, period_x=None, period_y=None, *,
                 _validate=True):
        self._init_walls(u_walls, v_walls, period_x, period_y)
        Nx, Ny = self.shape
        if vertex_images is None:
            V = np.empty((Nx + 1, Ny + 1, 2))
            V[..., 0] = self.u_bounds[:, None]
            V[..., 1] = self.v_bounds[None, :]
        else:
            V = np.array(vertex_images, dtype=float)
            if V.shape != (Nx + 1, Ny + 1, 2):
                raise ValueError(
                    f"TransfiniteMap: vertex_images must have shape "
                    f"{(Nx + 1, Ny + 1, 2)} (one (x, y) per grid vertex), got "
                    f"{V.shape}.")
        if not np.all(np.isfinite(V)):
            raise ValueError("TransfiniteMap: vertex_images must be finite.")
        self.vertex_images = V
        self.curved_edges = {}
        scale = max(self.period_x, self.period_y)
        for key, crv in dict(curved_edges or {}).items():
            try:
                kind, i, j = key
                i, j = int(i), int(j)
            except (TypeError, ValueError):
                raise ValueError(
                    f"TransfiniteMap: edge key {key!r} must be ('h', i, j) or "
                    f"('v', i, j).") from None
            if kind == "h":
                ok = 0 <= i < Nx and 0 <= j <= Ny
                Q = (i + 1, j)
            elif kind == "v":
                ok = 0 <= i <= Nx and 0 <= j < Ny
                Q = (i, j + 1)
            else:
                ok = False
            if not ok:
                raise ValueError(
                    f"TransfiniteMap: edge {key!r} is not an edge of the "
                    f"{Nx} x {Ny} grid ('h' edges run (i, j) -> (i + 1, j), "
                    f"'v' edges (i, j) -> (i, j + 1)).")
            if not callable(crv) or not callable(getattr(crv, "key", None)):
                raise TypeError(
                    f"TransfiniteMap: edge {key!r} must be an EdgeCurve "
                    f"(callable on s, with key()), got {type(crv).__name__}.")
            ends = crv(np.array([0.0, 1.0]))[0]
            for end, (vi, vj), nm in ((ends[0], (i, j), "s = 0"),
                                      (ends[1], Q, "s = 1")):
                d = float(np.max(np.abs(end - V[vi, vj])))
                if not d <= _EDGE_END_TOL * scale:
                    raise ValueError(
                        f"TransfiniteMap: the {nm} end of edge {key!r} is "
                        f"{tuple(end)}, but the image of vertex {(vi, vj)} "
                        f"is {tuple(V[vi, vj])} (mismatch {d:.3e}) -- the "
                        f"two cells sharing this edge would not meet (the "
                        f"map must be continuous across every grid line).")
            # the Jacobian is built from the curve's ANALYTIC derivative and
            # the positions from its value: they must agree, or the map's
            # Jacobian would not be the derivative of its positions (a user
            # curve with a derivative bug gives a silently wrong answer --
            # Phase B verifier D-1: a 10 % derivative bug moved R / T by
            # 3.2e-2 at M = 5 and was accepted)
            sp = np.linspace(0.1, 0.9, 5)
            hd = 1e-6
            fd = (crv(sp + hd)[0] - crv(sp - hd)[0]) / (2.0 * hd)
            an = crv(sp)[1]
            dmis = float(np.max(np.abs(fd - an)))
            if not dmis <= 1e-6 * max(scale, float(np.max(np.abs(an)))):
                raise ValueError(
                    f"TransfiniteMap: edge {key!r}: the curve's derivative is "
                    f"not d/ds of its value (mismatch {dmis:.3e} against a "
                    f"central difference) -- the map's Jacobian would not "
                    f"match its positions.")
            self.curved_edges[(kind, i, j)] = crv
        if _validate:
            # (the shape layer passes False to locate a fold itself and name
            # the shapes involved, then calls validate(n=12) -- the same
            # check, never skipped)
            self.validate(n=12)

    def edge(self, kind, i, j):
        """The curve of edge ``(kind, i, j)`` -- the given curve, or the
        straight :class:`Line` between its two vertex images."""
        c = self.curved_edges.get((kind, i, j))
        if c is not None:
            return c
        V = self.vertex_images
        Q = V[i + 1, j] if kind == "h" else V[i, j + 1]
        return Line(V[i, j], Q)

    @property
    def singular_vertices(self):
        """The cell corners where ``det J = 0``: a sorted list of
        ``(sx, sy, cu, cv)`` with ``cu, cv in {0, 1}`` the corner's side of
        cell ``(sx, sy)`` (0 = its lower wall).  A corner is singular when the
        cell's two edge tangents there are parallel (antiparallel for a
        closed smooth curve) to 1e-10 relative.

        A corner that is only NEARLY singular (a user curve whose tangents
        meet at an angle just above that tolerance -- ``det J`` small but
        positive) is NOT listed: its cell gets the tensor rule, whose moments
        then converge only algebraically, and the solver's node-count
        criterion WARNS at its cap with the moment error it reached (Phase B
        verifier note N-1).  The shipped primitives place their 45-degree
        points exactly, so this arises only for hand-built maps; the remedy
        is to make the corner exactly singular or to keep it well away from
        singular."""
        out = []
        Nx, Ny = self.shape
        one = np.array([0.0, 1.0])
        for sx in range(Nx):
            for sy in range(Ny):
                eB = self.edge("h", sx, sy)(one)[1]
                eT = self.edge("h", sx, sy + 1)(one)[1]
                eL = self.edge("v", sx, sy)(one)[1]
                eR = self.edge("v", sx + 1, sy)(one)[1]
                for (cu, cv), tu, tv in (((0, 0), eB[0], eL[0]),
                                         ((1, 0), eB[1], eR[0]),
                                         ((0, 1), eT[0], eL[1]),
                                         ((1, 1), eT[1], eR[1])):
                    cr = tu[0] * tv[1] - tu[1] * tv[0]
                    nrm = float(np.hypot(*tu) * np.hypot(*tv))
                    if abs(cr) <= 1e-10 * nrm:
                        out.append((sx, sy, cu, cv))
        return out

    def _blend(self, sx, sy, s, t, grid):
        """The Gordon-Hall formula and its two analytic derivatives on cell
        ``(sx, sy)`` at the local coordinates ``s``, ``t`` (in ``[0, 1]``):
        on the TENSOR grid ``s x t`` when ``grid`` (the :meth:`geom` layout,
        first index along ``u``), or at the POINTS ``(s[k], t[k])``
        otherwise (:meth:`geom_points`).  One formula, two broadcasts."""
        Bv, Bd = self.edge("h", sx, sy)(s)          # bottom (t = 0)
        Tv, Td = self.edge("h", sx, sy + 1)(s)      # top    (t = 1)
        Lv, Ld = self.edge("v", sx, sy)(t)          # left   (s = 0)
        Rv, Rd = self.edge("v", sx + 1, sy)(t)      # right  (s = 1)
        if grid:
            S, Tt = s[:, None, None], t[None, :, None]
            Bv, Bd, Tv, Td = (a[:, None, :] for a in (Bv, Bd, Tv, Td))
            Lv, Ld, Rv, Rd = (a[None, :, :] for a in (Lv, Ld, Rv, Rd))
        else:
            S, Tt = s[:, None], t[:, None]
        Vi = self.vertex_images
        P00, P10 = Vi[sx, sy], Vi[sx + 1, sy]
        P01, P11 = Vi[sx, sy + 1], Vi[sx + 1, sy + 1]
        Phi = ((1 - Tt) * Bv + Tt * Tv + (1 - S) * Lv + S * Rv
               - ((1 - S) * (1 - Tt) * P00 + S * (1 - Tt) * P10
                  + (1 - S) * Tt * P01 + S * Tt * P11))
        Ps = ((1 - Tt) * Bd + Tt * Td - Lv + Rv
              - ((1 - Tt) * (P10 - P00) + Tt * (P11 - P01)))
        Pt = (-Bv + Tv + (1 - S) * Ld + S * Rd
              - ((1 - S) * (P01 - P00) + S * (P11 - P10)))
        du = self.u_bounds[sx + 1] - self.u_bounds[sx]
        dv = self.v_bounds[sy + 1] - self.v_bounds[sy]
        return (Phi[..., 0], Phi[..., 1], Ps[..., 0] / du, Pt[..., 0] / dv,
                Ps[..., 1] / du, Pt[..., 1] / dv)

    def _local(self, sx, sy, U, V):
        u0, u1 = self.u_bounds[sx], self.u_bounds[sx + 1]
        v0, v1 = self.v_bounds[sy], self.v_bounds[sy + 1]
        s = (np.atleast_1d(np.asarray(U, dtype=float)) - u0) / (u1 - u0)
        t = (np.atleast_1d(np.asarray(V, dtype=float)) - v0) / (v1 - v0)
        return s, t

    def geom(self, sx, sy, U, V):
        s, t = self._local(sx, sy, U, V)
        return self._blend(sx, sy, s, t, True)

    def geom_points(self, sx, sy, U, V):
        s, t = self._local(sx, sy, U, V)
        return self._blend(sx, sy, s.ravel(), t.ravel(), False)

    def _key(self):
        return (self.vertex_images.tobytes(),
                tuple((k, self.curved_edges[k].key())
                      for k in sorted(self.curved_edges)))


class RefinedMap(CellMap):
    """A FINER wall grid on an unchanged geometry: the h-refinement of a map.

    What it is for.  The staggered basis needs a SQUARE wall grid
    (``Nx == Ny``), and a shape layout is rarely square (a circle and a
    rectangle side by side can need five ``u`` walls and three ``v`` walls).
    Adding a wall to square it up must not move any material boundary, so
    the added wall is a SUBDIVISION of the base map's cells: every fine cell
    lies inside one cell of ``base`` and the map there IS the base cell's own
    blend, evaluated at the fine cell's points.  Nothing physical changes --
    the image of every base grid line, curved or straight, is exactly what it
    was -- and only the solver's resolution is redistributed (Phase C of the
    curved-cell plan; the plan's "a wall that crosses another shape's curved
    macro-cell subdivides it, and each sub-cell takes the macro-cell's blend
    evaluated on its sub-rectangle").

    Parameters
    ----------
    base : CellMap
        The map to refine (in practice a :class:`TransfiniteMap`).
    u_walls, v_walls : increasing ``(N + 1,)`` arrays
        The fine boundary arrays; each must CONTAIN every boundary of the base
        grid (to ``1e-13`` of the period -- a base wall that is not a fine
        wall would put a kink of the map inside a fine cell, where the
        quadrature assumes an analytic map, and is refused).

    The singular vertices of ``base`` are carried over to the fine cells that
    own them, so the solver's corner (Duffy) rule still runs exactly where the
    map pinches.  The fingerprint is the base's plus the fine walls."""

    def __init__(self, base, u_walls, v_walls):
        if not isinstance(base, CellMap):
            raise TypeError(f"RefinedMap: base must be a CellMap, got "
                            f"{type(base).__name__}.")
        self.base = base
        self._init_walls(u_walls, v_walls, base.period_x, base.period_y)
        self._owner_u = self._owners(self.u_bounds, base.u_bounds,
                                     self.period_x, "u")
        self._owner_v = self._owners(self.v_bounds, base.v_bounds,
                                     self.period_y, "v")
        self.validate()

    @staticmethod
    def _owners(fine, coarse, period, axis):
        """Index of the base segment holding each fine segment; refuses a
        base wall that is not a fine wall."""
        tol = 1e-13 * float(period)
        for c in coarse:
            if float(np.min(np.abs(fine - c))) > tol:
                raise ValueError(
                    f"RefinedMap: the base {axis}-wall {c!r} is not a wall of "
                    f"the fine grid -- a refinement must keep every base "
                    f"wall (the map may kink there).")
        mids = 0.5 * (fine[:-1] + fine[1:])
        return np.clip(np.searchsorted(coarse, mids) - 1, 0, coarse.size - 2)

    def geom(self, sx, sy, U, V):
        return self.base.geom(int(self._owner_u[sx]), int(self._owner_v[sy]),
                              U, V)

    def geom_points(self, sx, sy, U, V):
        return self.base.geom_points(int(self._owner_u[sx]),
                                     int(self._owner_v[sy]), U, V)

    @property
    def singular_vertices(self):
        """The base's singular vertices, re-indexed onto the fine cells that
        own them (the fine cell inside the base cell with that corner)."""
        out = []
        tol = 1e-13 * max(self.period_x, self.period_y)
        bu, bv = self.base.u_bounds, self.base.v_bounds
        for bsx, bsy, cu, cv in self.base.singular_vertices:
            u = bu[bsx + cu]
            v = bv[bsy + cv]
            iu = int(np.argmin(np.abs(self.u_bounds - u)))
            iv = int(np.argmin(np.abs(self.v_bounds - v)))
            if (abs(self.u_bounds[iu] - u) > tol
                    or abs(self.v_bounds[iv] - v) > tol):
                continue                      # unreachable: walls are kept
            out.append((iu - cu, iv - cv, cu, cv))
        return sorted(out)

    def _key(self):
        return ("RefinedMap", type(self.base).__name__,
                self.base.fingerprint)


# =========================================================================== #
# Phase B gate maps (private).  The four layouts the build gates run on --
# lifted from the planning probe's builders (``circle_map_3x3``,
# ``circle_map_5x5``, ``fillet_map_5x5`` in
# ``validation/probe_pmm2d_curved/_curved_scratch.py``) plus the ellipse and
# the sinusoidal stripe of gate B9.  Phase C wraps them as public shape
# primitives; until then they are the expert-level way to build the maps.
# =========================================================================== #
_DEG = np.pi / 180.0


def _circle_map_3x3(period, radius, center=None):
    """``(TransfiniteMap, walls)`` -- a circle of ``radius`` in the square
    cell ``[0, period]^2`` on a 3 x 3 wall grid.  The walls pass through the
    circle's four 45-degree points (``u, v = c -+ r / sqrt 2``), every grid
    vertex is identity-mapped, and only the middle cell's four edges are
    90-degree arcs, so the middle cell IS the disk.  ``det J = 0`` at the
    middle cell's four corners (the four singular vertices)."""
    P = float(period)
    c = (P / 2.0, P / 2.0) if center is None else tuple(float(v) for v in
                                                         center)
    h = float(radius) / np.sqrt(2.0)
    w = np.array([0.0, c[0] - h, c[0] + h, P])
    wv = np.array([0.0, c[1] - h, c[1] + h, P])
    r = float(radius)
    curved = {("h", 1, 1): Arc(c, r, 225 * _DEG, 315 * _DEG),
              ("h", 1, 2): Arc(c, r, 135 * _DEG, 45 * _DEG),
              ("v", 1, 1): Arc(c, r, 225 * _DEG, 135 * _DEG),
              ("v", 2, 1): Arc(c, r, -45 * _DEG, 45 * _DEG)}
    return TransfiniteMap(w, wv, None, curved), w


def _ellipse_map_3x3(period, semi_axes, center=None):
    """``(TransfiniteMap, (u_walls, v_walls))`` -- the axis-aligned ellipse
    analogue of :func:`_circle_map_3x3`: walls through the PARAMETRIC
    45-degree points ``c -+ (a, b) / sqrt 2``, the middle cell's four edges
    90-degree :class:`EllipseArc` s; four singular vertices."""
    P = float(period)
    a, b = (float(v) for v in semi_axes)
    c = (P / 2.0, P / 2.0) if center is None else tuple(float(v) for v in
                                                         center)
    hu, hv = a / np.sqrt(2.0), b / np.sqrt(2.0)
    wu = np.array([0.0, c[0] - hu, c[0] + hu, P])
    wv = np.array([0.0, c[1] - hv, c[1] + hv, P])
    ax = (a, b)
    curved = {("h", 1, 1): EllipseArc(c, ax, 225 * _DEG, 315 * _DEG),
              ("h", 1, 2): EllipseArc(c, ax, 135 * _DEG, 45 * _DEG),
              ("v", 1, 1): EllipseArc(c, ax, 225 * _DEG, 135 * _DEG),
              ("v", 2, 1): EllipseArc(c, ax, -45 * _DEG, 45 * _DEG)}
    return TransfiniteMap(wu, wv, None, curved), (wu, wv)


def _circle_map_5x5(period, radius, inner=0.5):
    """``(TransfiniteMap, walls)`` -- the same circle on a 5 x 5 wall grid:
    the DISK is the inner 3 x 3 block, its centre cell a straight-edged square
    of half-size ``inner * r / sqrt 2``; the loop's corner vertices sit at the
    45-degree points (identity-mapped) and its side vertices on the circle
    straight above / beside the inner walls, so every loop edge is a pure
    arc.  The same four singular vertices as the 3 x 3 map; the extra cells
    give the disk's interior a better-shaped parametrisation."""
    P = float(period)
    r = float(radius)
    c = np.array([P / 2.0, P / 2.0])
    a1 = P / 2.0 - r / np.sqrt(2.0)
    h = inner * r / np.sqrt(2.0)
    a2 = P / 2.0 - h
    w = np.array([0.0, a1, a2, P - a2, P - a1, P])
    phi = np.arcsin(h / r)

    def circ(th):
        return (c[0] + r * np.cos(th), c[1] + r * np.sin(th))

    def img(i, j):
        x, y = w[i], w[j]
        if i in (1, 4) and j in (1, 4):
            return (x, y)                         # 45-degree points
        if j in (1, 4) and i in (2, 3):           # bottom / top loop side
            sgn = -1 if i == 2 else 1
            base = 270 * _DEG if j == 1 else 90 * _DEG
            return circ(base + (sgn * phi if j == 1 else -sgn * phi))
        if i in (1, 4) and j in (2, 3):           # left / right loop side
            sgn = -1 if j == 2 else 1
            return circ(180 * _DEG - sgn * phi if i == 1 else sgn * phi)
        return (x, y)
    verts = np.array([[img(i, j) for j in range(6)] for i in range(6)])
    curved = {}
    for i in (1, 2, 3):
        for j in (1, 4):
            curved[("h", i, j)] = Arc.through(verts[i, j], verts[i + 1, j], c)
    for j in (1, 2, 3):
        for i in (1, 4):
            curved[("v", i, j)] = Arc.through(verts[i, j], verts[i, j + 1], c)
    return TransfiniteMap(w, w, verts, curved), w


def _fillet_map_5x5(period, half, rf):
    """``(TransfiniteMap, walls)`` -- the square pillar ``[P/2 - half,
    P/2 + half]^2`` with its four corners rounded to radius ``rf > 0``, on a
    5 x 5 wall grid whose walls pass through the 45-degree fillet points
    (identity-mapped) and the arc/line TANGENCY points, so every cell edge is
    a pure arc or a pure line and the map is analytic inside every cell.  The
    pillar is the inner 3 x 3 block; the fillet's own segment is
    ``rf (1 - 1 / sqrt 2)`` wide."""
    P = float(period)
    lo = P / 2.0 - half
    hi = P / 2.0 + half
    a = lo + rf * (1.0 - 1.0 / np.sqrt(2.0))
    b = lo + rf
    w = np.array([0.0, a, b, P - b, P - a, P])

    def img(i, j):
        x, y = w[i], w[j]
        if i in (1, 4) and j in (1, 4):
            return (x, y)                          # 45-degree fillet points
        if j in (1, 4) and i in (2, 3):
            return (x, lo if j == 1 else hi)       # bottom/top tangency
        if i in (1, 4) and j in (2, 3):
            return (lo if i == 1 else hi, y)       # left/right tangency
        return (x, y)
    verts = np.array([[img(i, j) for j in range(6)] for i in range(6)])
    cBL, cBR, cTL, cTR = (b, b), (P - b, b), (b, P - b), (P - b, P - b)
    curved = {
        ("h", 1, 1): Arc(cBL, rf, 225 * _DEG, 270 * _DEG),
        ("h", 3, 1): Arc(cBR, rf, 270 * _DEG, 315 * _DEG),
        ("h", 1, 4): Arc(cTL, rf, 135 * _DEG, 90 * _DEG),
        ("h", 3, 4): Arc(cTR, rf, 90 * _DEG, 45 * _DEG),
        ("v", 1, 1): Arc(cBL, rf, 225 * _DEG, 180 * _DEG),
        ("v", 1, 3): Arc(cTL, rf, 180 * _DEG, 135 * _DEG),
        ("v", 4, 1): Arc(cBR, rf, 315 * _DEG, 360 * _DEG),
        ("v", 4, 3): Arc(cTR, rf, 0 * _DEG, 45 * _DEG),
    }
    return TransfiniteMap(w, w, verts, curved), w


def _sine_stripe_map_3x3(period, x_left, x_right, amplitude, v_walls=None):
    """``(TransfiniteMap, walls)`` -- a ridge of constant width bounded by
    the two in-phase sinusoidal walls ``x = x_left + A sin(2 pi y / p)`` and
    ``x = x_right + A sin(2 pi y / p)``, on a 3 x 3 grid with ``u`` walls
    ``{0, x_left, x_right, p}`` and ``v`` walls ``v_walls`` (default the
    uniform thirds).  The two interior vertical grid lines ARE the sinusoids
    (one :class:`Sinusoid` piece per cell edge), every horizontal edge is a
    straight line, so the map is ``x = blend``, ``y = v`` -- analytic, with
    ``det J > 0`` everywhere (no singular vertex) while ``|A| < x_left`` and
    ``|A| < p - x_right``.  The ridge is the middle COLUMN of cells."""
    P = float(period)
    A = float(amplitude)
    wu = np.array([0.0, float(x_left), float(x_right), P])
    wv = (np.array([0.0, P / 3.0, 2.0 * P / 3.0, P]) if v_walls is None
          else np.asarray(v_walls, dtype=float))
    k = 2.0 * np.pi / P
    verts = np.empty((4, wv.size, 2))
    for i in range(4):
        for j in range(wv.size):
            dx = A * np.sin(k * wv[j]) if i in (1, 2) else 0.0
            verts[i, j] = (wu[i] + dx, wv[j])
    curved = {}
    for i in (1, 2):
        for j in range(wv.size - 1):
            curved[("v", i, j)] = Sinusoid(wu[i], A, P, wv[j], wv[j + 1])
    return TransfiniteMap(wu, wv, verts, curved), (wu, wv)
