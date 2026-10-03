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
:class:`IdentityMap` and :class:`SeparableStretch` are the two maps this
phase ships (the curved shape maps come later).

Validation (``CellMap.validate``)
---------------------------------
* **Lattice periodicity.**  ``Phi(u + p_x, v) = Phi(u, v) + (p_x, 0)`` and
  ``Phi(u, v + p_y) = Phi(u, v) + (0, p_y)``, and the Jacobian is periodic.
  This is what keeps the solver's Bloch boundary condition (the phase ``tau``
  that glues the cell's two edges) unchanged under the map, and what makes the
  image of the ``(u, v)`` cell one full period of the physical lattice.
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

__all__ = ["CellMap", "IdentityMap", "SeparableStretch", "SineStretch"]

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
        jscale = max(1.0, max(float(np.max(np.abs(t))) for t in a[2:]))
        djac = max(float(np.max(np.abs(tb - ta))) for ta, tb in
                   zip(a[2:], b[2:]))
        if dpos > tol * scale or djac > tol * jscale:
            raise ValueError(
                f"{type(self).__name__}: the map must be lattice-periodic -- "
                f"Phi(u + p_x, v) = Phi(u, v) + (p_x, 0) and Phi(u, v + p_y) "
                f"= Phi(u, v) + (0, p_y) with a periodic Jacobian -- so that "
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

    def _key(self):
        return (None if self.fx is None else self.fx.key(),
                None if self.fy is None else self.fy.key())
