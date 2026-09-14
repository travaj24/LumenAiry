"""RCWA analytic-shape GEOMETRY PREDICATES: the support function, the exact
pair-overlap test, the unit-disk hit tests and the y-invariance test.

A leaf of the RCWA package: it imports ``numpy`` and nothing else from the
library, so it can be read, tested and reasoned about without the S-matrix
algebra, the eigensolvers or the BLAS controls in ``_core``.  ``_core``
re-exports every name below, so
``lumenairy.elements.rcwa._core._shapes_overlap`` and
``lumenairy.elements.rcwa._shapes_overlap`` keep resolving exactly as they did.

WHAT BELONGS HERE.  Everything that answers a question about shapes on a
rectangular period in PLANE GEOMETRY, with no electromagnetics and no policy in
it: does this pair overlap, does this ellipse meet this disk, does this list
vary along y.

WHAT DOES NOT.  ``_core._validate_shapes`` composes these predicates into the
public refusals -- which malformed entry raises what, which tolerance is
forgiven -- and stays in ``_core``, both because that is policy rather than
geometry and because it must resolve ``_shapes_overlap`` through ``_core``'s
own globals: ``tests/unit/test_niche_audit_w9_overlap_exact.py`` substitutes a
counting wrapper there to prove the exact predicate is actually reached.
``_core._validate_cell_sampling`` (the RASTERISED-cell path) likewise stays
beside the Fourier machinery it guards.
"""
from __future__ import annotations

import numpy as np

# ONE-SIDED forgiveness window of the overlap guard, as a fraction of the
# smaller period (audit W8, kept in W9).  This is a deliberate TOLERANCE, not a
# limit of the predicate: since W9 the pair predicates are EXACT algebra (the
# measured intrinsic floor is ~1e-14 of a period, see
# ``tests/unit/test_niche_audit_w9_overlap_exact.py``), so this number is the
# ONLY blindness left, and it is here so that a layout whose centres were built
# by float arithmetic (trig, cumulative pitches) is not rejected for a
# sub-picometre numerical graze.  Lower it -- ``_shapes_overlap(...,
# tol_frac=...)`` -- for a strict consumer that wants every real overlap caught.
_OVERLAP_SLACK_FRAC = 1e-6


def _shape_support(kind, sx, sy, ux, uy):
    """Support function ``h(u) = max_{r in shape} r.u`` of a shape CENTRED at
    the origin, evaluated at the unit directions ``(ux, uy)``.  ``(sx, sy)``
    are the HALF-widths of a ``rectangle`` and the semi-axes of a
    ``disk`` / ``ellipse`` -- i.e. its bounding-box half-extents either way.

    Retained as the INDEPENDENT cross-check of the exact predicates: before
    audit W9 :func:`_shapes_overlap` decided every curved pair by scanning this
    over 4096 directions (the ``_OVERLAP_DIRS`` constant, removed with the
    scan), and the W9 pins re-implement that scan against this function and
    measure the two verdicts against each other on 20000 random pairs."""
    if kind == "rectangle":
        return sx * np.abs(ux) + sy * np.abs(uy)
    return np.hypot(sx * ux, sy * uy)


_ELLIPSE_BISECT_STEPS = 64      # fixed (deterministic) bisection depth


def _point_ellipse_distance(u, v, p, q):
    """EXACT Euclidean distance from a point ``(u, v)`` STRICTLY OUTSIDE the
    axis-aligned ellipse ``(x/p)^2 + (y/q)^2 = 1`` (centred at the origin) to
    that ellipse (audit W9).

    The foot of the perpendicular is ``(p^2 u / (t + p^2), q^2 v / (t + q^2))``
    for the unique ``t >= 0`` solving the monotone

        ``F(t) = (p u / (t + p^2))^2 + (q v / (t + q^2))^2 - 1 = 0``

    (Eberly's reduction of the distance quartic).  ``F`` is STRICTLY DECREASING
    on ``t >= 0`` with ``F(0) > 0`` for an outside point, and ``F(t) < 0``
    whenever ``t > sqrt(p^2 u^2 + q^2 v^2)`` (because ``t + p^2 > t``), so
    ``[0, sqrt(p^2 u^2 + q^2 v^2)]`` is a PROVEN bracket -- no unbracketed
    iteration, and a FIXED ``_ELLIPSE_BISECT_STEPS`` halvings keep the result
    bit-reproducible across platforms (bisection uses only ``+``, ``*``, ``/``
    and comparisons, all IEEE-exact operations).  ``t_hi * 2^-64`` is ~1e-19
    relative, far below the guard's tolerance.

    Callers MUST have established ``(u/p)^2 + (v/q)^2 > 1`` (outside); an
    inside point needs no distance -- it is an overlap outright."""
    au, av = abs(float(u)), abs(float(v))
    p2, q2 = float(p) * float(p), float(q) * float(q)
    lo = 0.0
    hi = float(np.hypot(float(p) * au, float(q) * av))
    for _ in range(_ELLIPSE_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        fx = float(p) * au / (mid + p2)
        fy = float(q) * av / (mid + q2)
        if fx * fx + fy * fy - 1.0 > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return float(np.hypot(au * t / (t + p2), av * t / (t + q2)))


def _ellipse_hits_unit_disk(dx, dy, p, q):
    """``True`` when the axis-aligned ellipse of semi-axes ``(p, q)`` centred at
    ``(dx, dy)`` meets the CLOSED unit disk at the origin -- with TANGENCY
    excluded (strict), the one-sided contract.  Exact: the two meet iff the
    distance from the origin to the ellipse REGION is ``< 1``, and the origin is
    either inside the ellipse (distance 0) or outside it, where
    :func:`_point_ellipse_distance` is exact."""
    if p <= 0.0 or q <= 0.0:
        return False
    # Re-centre on the ellipse: the origin of the disk sits at (-dx, -dy).
    if (dx / p) ** 2 + (dy / q) ** 2 <= 1.0:
        return True                        # disk centre inside the ellipse
    return bool(_point_ellipse_distance(-dx, -dy, p, q) < 1.0)


def _shapes_overlap(a, b, period_x, period_y, *, tol_frac=None):
    """``True`` when two shape descriptors ``(kind, sx, sy, (cx, cy))`` overlap
    on the PERIODIC cell (audit W8 2026-07-27; made EXACT in audit W9).

    ``d`` is the MINIMAL periodic image, which is sufficient: the Minkowski sum
    of two axis-aligned centrally-symmetric convex bodies is convex and
    reflection-symmetric about both axes, so shrinking ``|dx|`` (or ``|dy|``) can
    only move ``d`` INTO the sum -- if any periodic image overlaps, the minimal
    one does.  A bounding-box separation is tried first: it is cheap, it is EXACT
    for a rectangle pair (the box IS the shape), and it keeps the algebra off all
    but the genuinely-close pairs.

    EXACT PAIR PREDICATES (audit W9; ``sx``/``sy`` are half-widths for a
    ``rectangle`` and semi-axes for a ``disk``/``ellipse``, and every shape kind
    is AXIS-ALIGNED -- the shape dicts carry no rotation entry, and neither do
    the form factors that read them):

    * rect / rect -- per-axis interval overlap, i.e. the bounding-box test that
      already ran;
    * disk / disk -- centre distance against the radius sum;
    * rect / ellipse (a disk is the ``p == q`` ellipse) -- scale by
      ``(1/p, 1/q)``: an axis-aligned scaling maps the ellipse to the UNIT DISK
      and keeps the rectangle axis-aligned, so the test is the closest point of
      an axis-aligned box to the origin;
    * ellipse / ellipse -- scale by the FIRST ellipse's semi-axes: it becomes the
      unit disk and (both being axis-aligned) the second stays an axis-aligned
      ellipse, reducing every curved pair to POINT-ELLIPSE distance
      (:func:`_point_ellipse_distance`, a bracketed monotone root -- the
      distance quartic, solved without any unbracketed iteration).

    This replaces the pre-W9 4096-direction scan of :func:`_shape_support`,
    whose separating-axis maximum was UNDER-estimated by up to ~2e-7 of a period
    (an approximation inside the predicate).  What remains is a single explicit
    ONE-SIDED tolerance: BOTH shapes are eroded by ``tol/2`` before the test --
    evenly, so the verdict cannot depend on the pair ORDER, and summing to
    exactly the historical ``dist < ra + rb - tol`` for two disks and to the
    bounding-box test for two boxes.  A pair that TOUCHES or is gapped is
    therefore never reported (tangent / abutting shapes stay LEGAL: their
    intersection has measure zero and superposition is still right) while any
    overlap deeper than ``tol`` always is.

    ``tol_frac`` overrides :data:`_OVERLAP_SLACK_FRAC` (fraction of the smaller
    period) for a caller that wants the exactness rather than the forgiveness.
    MEASURED with the window set below the depth: overlaps of 1e-8, 1e-10, 1e-12
    and 1e-14 of a period are all resolved, and tangency at exactly 0 is not
    reported at any of them.  (One consequence of the even erosion: a shape
    whose semi-axis is BELOW ``tol/2`` lies entirely inside its own forgiveness
    window and is reported disjoint.)"""
    ka, sax, say, ca = a
    kb, sbx, sby, cb = b
    frac = _OVERLAP_SLACK_FRAC if tol_frac is None else float(tol_frac)
    tol = frac * min(float(period_x), float(period_y))
    sax, say, sbx, sby = float(sax), float(say), float(sbx), float(sby)
    dx = float(((cb[0] - ca[0]) + 0.5 * period_x) % period_x - 0.5 * period_x)
    dy = float(((cb[1] - ca[1]) + 0.5 * period_y) % period_y - 0.5 * period_y)
    if abs(dx) > sax + sbx - tol or abs(dy) > say + sby - tol:
        return False                       # bounding boxes already separate
    if ka == "rectangle" and kb == "rectangle":
        return True                        # for two boxes that IS the test
    # Erode BOTH shapes by tol/2 -- the one-sided window, split evenly so the
    # verdict cannot depend on the pair ORDER (for two disks this is exactly the
    # historical ``dist < ra + rb - tol``, and for two boxes exactly the
    # bounding-box test above).
    h = 0.5 * tol
    eax, eay, ebx, eby = sax - h, say - h, sbx - h, sby - h
    if min(eax, eay, ebx, eby) <= 0.0:
        return False                       # eroded away: nothing left to hit
    if ka != "rectangle" and kb != "rectangle":
        if eax == eay and ebx == eby:      # disk / disk: closed form
            # bool(...): a numpy-scalar semi-axis would otherwise leak a
            # np.bool_ out of the predicate, and callers test it with ``is``.
            return bool(float(np.hypot(dx, dy)) < float(eax) + float(ebx))
        # scale the FIRST ellipse to the unit disk; the second stays an
        # axis-aligned ellipse, so the pair reduces to point-ellipse distance
        return _ellipse_hits_unit_disk(dx / eax, dy / eay,
                                       ebx / eax, eby / eay)
    if ka == "rectangle":                  # rect / ellipse -> scale by B
        return _box_hits_unit_disk(dx / ebx, dy / eby, eax / ebx, eay / eby)
    # ellipse / rect -> scale by A
    return _box_hits_unit_disk(dx / eax, dy / eay, ebx / eax, eby / eay)


def _box_hits_unit_disk(cx, cy, hx, hy):
    """``True`` when the axis-aligned box of half-widths ``(hx, hy)`` centred at
    ``(cx, cy)`` meets the unit disk at the origin, TANGENCY EXCLUDED.  Exact:
    the closest point of the box to the origin is the per-axis clamp, so the
    distance is ``hypot(max(0, |cx| - hx), max(0, |cy| - hy))``."""
    gx = max(0.0, abs(float(cx)) - float(hx))
    gy = max(0.0, abs(float(cy)) - float(hy))
    return bool(float(np.hypot(gx, gy)) < 1.0)


def _shapes_y_varying(shapes, period_y):
    """Index of the first shape in ``shapes`` that VARIES along y, or ``None``
    when the whole list is y-INVARIANT (an empty list included -- a uniform
    background).

    THE SINGLE definition of "y-invariant shape list", shared by the
    ``n_orders_y = 0`` RAISE in :func:`_validate_shapes` (the explicitly-2-D
    entry points) and the 1-D-stack ``RCWAYAverageWarning`` DIAGNOSTIC in
    :func:`~lumenairy.elements.rcwa.stack._warn_if_shapes_y_averaged`, so the
    two can never reach different verdicts about the same shape list -- the
    no-divergence contract commit ``809314c`` pinned for the pixel path
    (``_warn_if_y_averaged`` vs the M8 ``strict_y`` branch), carried over to the
    analytic flavour in audit W8.

    Only a RECTANGLE spanning the full ``period_y`` is y-invariant: it tiles y,
    so its form factor's ``sinc(l)`` kills every ``l != 0`` coefficient
    EXACTLY (measured: such a stripe solved with ``n_orders_y = 0`` reproduces
    the y-resolved solve to 2.9e-16).  A disk, an ellipse or a shorter
    rectangle all carry real y structure, and retaining no y-harmonic would
    silently solve their y-AVERAGE instead (measured on a disk: R00 = 0.054846
    against the y-resolved 0.006897, energy closure 4.4e-16).  Geometry is
    assumed already validated by :func:`_validate_shapes`; both callers run
    immediately after it."""
    for i, sh in enumerate(shapes):
        if sh.get("shape") == "rectangle":
            wy = float(sh["size"][1])
            if wy >= float(period_y) * (1.0 - 1e-9):
                continue                      # a full-height stripe IS y-flat
        return i
    return None


