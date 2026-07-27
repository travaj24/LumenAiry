"""Wave-9 pins, part D -- EXACT pair predicates in the analytic-shape OVERLAP
guard (``lumenairy/elements/rcwa/_core.py``: :func:`_shapes_overlap`).

W8 added the guard with a 4096-direction scan of the support functions
(:func:`_shape_support`) -- a discretization of the separating-axis criterion,
whose maximum it UNDER-estimates by up to ~2e-7 of a period.  That
approximation lived INSIDE the predicate, so the guard's blindness was the
entanglement of two unrelated numbers: the deliberate one-sided slack
(``1e-6 * period``) and the scan's own error.  W9 replaces the scan with exact
algebra, leaving the slack as the ONLY blindness:

  rect / rect     -- per-axis interval overlap (already exact: the box IS the
                     shape, and the bounding-box pre-filter is that test)
  disk / disk     -- centre distance vs the radius sum
  rect / ellipse  -- axis-scale the ellipse to the UNIT DISK (an axis-aligned
                     scaling keeps the rectangle axis-aligned) -> closest point
                     of a box to the origin
  ellipse/ellipse -- axis-scale the FIRST ellipse to the unit disk; the second
                     stays axis-aligned, so every curved pair reduces to
                     POINT-ELLIPSE distance: the distance quartic solved as a
                     BRACKETED monotone root (Eberly's reduction), never an
                     unbracketed iteration

Shapes are AXIS-ALIGNED throughout -- the shape dicts carry no rotation entry
and neither ``_validate_shapes`` nor ``_shape_form_factor`` reads one -- so the
axis-scaling reduction covers every pair the suite can build.

WHAT IT BUYS, MEASURED.  The shipped tolerance is UNCHANGED
(``_OVERLAP_SLACK_FRAC = 1e-6``): it is a deliberate forgiveness for layouts
whose centres came out of float arithmetic, not blindness, and every W8 pin
(including the recorded 1e-13 m / 1e-11 m switch-over window) is reproduced
bit-for-bit -- 20000 random pairs agree with the old scan, 0 disagreements.
What changes is that the tolerance can now be BELIEVED: at ``tol_frac <= 1e-8``
the old scan reported 406-735 FALSE POSITIVES per 3000 exactly-tangent /
gapped LEGAL pairs (its own under-estimate exceeding the window), while the
exact predicate reports 0 at every tolerance down to 1e-12 -- so the detection
floor is now the tolerance itself, measured to resolve overlaps of 1e-8, 1e-10,
1e-12 and 1e-14 of a period while never flagging tangency at exactly 0.
Faster, too: 1024 disks 51.8 ms against the W8-recorded 81 ms.

MEASURED PRE-FIX CHECK: 19 pins here (14 functions, one 6-way parametrized), of
which 13 FAIL on a clean ``a3b185c`` worktree -- the ``tol_frac`` keyword, the
``_OVERLAP_SLACK_FRAC`` constant and ``_point_ellipse_distance`` do not exist
there, and the tightened-window claims are exactly what the old scan cannot
support.  The 6 that pass are the documented NON-DISCRIMINATORS -- the CONTRACT
the exactness is built on, which W9 must not move:

  test_w9d_shipped_window_is_unchanged        (the W8 window, bit-for-bit)
  test_w9d_tangency_and_abutment_stay_legal   (the one-sided contract)
  test_w9d_verdict_is_symmetric_in_pair_order
  test_w9d_periodic_seam_is_still_seen
  test_w9d_performance_envelope               (the 81 ms envelope)
  test_w9d_exact_predicate_agrees_with_the_old_scan_at_the_shipped_window
      (a tautology pre-fix -- it compares the old scan with itself; post-fix it
      is the independent 20000-pair cross-check)

ONE DELIBERATE BEHAVIOUR CHANGE, recorded by
:func:`test_w9d_degenerate_and_extreme_geometry_is_handled`: a shape whose
semi-axis is BELOW the tolerance is now eroded away and reported disjoint (it
lies entirely inside its own forgiveness window), where the pre-W9 scan reported
an overlap.  At 0.06 pm of extent that geometry is far below anything the form
factors resolve; it is a consequence of making the window the only blindness.
"""
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa._core import (
    _shape_support,
    _shapes_overlap,
    _validate_shapes,
)

_P = 0.6e-6                      # the W8 shapes-audit cell, so the windows match
_D, _WL = 0.20e-6, 0.55e-6
_R = 0.08e-6


def _tol_overlap(a, b, tol_frac):
    """``_shapes_overlap`` at an explicit tolerance, with a clear failure on a
    pre-W9 tree (where the keyword does not exist)."""
    try:
        return _shapes_overlap(a, b, _P, _P, tol_frac=tol_frac)
    except TypeError as exc:       # pragma: no cover -- pre-fix tree only
        raise AssertionError(
            "pre-W9 tree: _shapes_overlap has no tol_frac keyword, so the "
            "guard's blindness cannot be separated from its predicate "
            f"({exc})") from None


def _old_scan(a, b, tol_frac):
    """The PRE-W9 predicate, verbatim -- the independent cross-check."""
    ka, sax, say, ca = a
    kb, sbx, sby, cb = b
    tol = tol_frac * _P
    dx = ((cb[0] - ca[0]) + 0.5 * _P) % _P - 0.5 * _P
    dy = ((cb[1] - ca[1]) + 0.5 * _P) % _P - 0.5 * _P
    if abs(dx) > sax + sbx - tol or abs(dy) > say + sby - tol:
        return False
    if ka == "rectangle" and kb == "rectangle":
        return True
    th = 2.0 * np.pi * np.arange(4096) / 4096
    ux, uy = np.cos(th), np.sin(th)
    gap = float(np.max(dx * ux + dy * uy
                       - _shape_support(ka, sax, say, ux, uy)
                       - _shape_support(kb, sbx, sby, ux, uy)))
    return gap < -tol


def _quad_family(gap):
    """The four pair KINDS at a signed separation ``gap`` along x (negative =
    overlapping by that depth): disk/disk, ellipse/ellipse, rect/ellipse and
    ellipse/rect -- the last two check that eroding the FIRST vs the SECOND
    shape cannot change the verdict."""
    a = ("disk", _R, _R, (0.15e-6, 0.3e-6))
    b = ("disk", _R, _R, (0.15e-6 + 2 * _R + gap, 0.3e-6))
    ea = ("ellipse", 0.10e-6, _R, (0.15e-6, 0.3e-6))
    eb = ("ellipse", _R, 0.05e-6, (0.15e-6 + 0.10e-6 + _R + gap, 0.3e-6))
    ra = ("rectangle", 0.10e-6, _R, (0.15e-6, 0.3e-6))
    return ((a, b), (ea, eb), (ra, eb), (eb, ra))


def _rand_shape(rng, kind=None):
    kind = kind or str(rng.choice(("rectangle", "disk", "ellipse")))
    s = rng.uniform(0.03, 0.22, 2) * 1e-6
    if kind == "disk":
        s[1] = s[0]
    return (kind, float(s[0]), float(s[1]),
            (float(rng.uniform(0, _P)), float(rng.uniform(0, _P))))


# ===========================================================================
# W9-D1 -- the contract W9 must not move (NON-DISCRIMINATORS)
# ===========================================================================

def test_w9d_shipped_window_is_unchanged():
    """NON-DISCRIMINATOR (passes pre-fix).  The DEFAULT guard behaves exactly as
    W8 recorded it: the one-sided switch-over sits between 1e-13 m (not
    reported) and 1e-11 m (reported), identically for disk/disk,
    ellipse/ellipse and rectangle/ellipse.  This is the W8 pin restated as a
    W9 regression guard -- the exact predicates must not tighten the SHIPPED
    tolerance, because a layout whose centres came out of trig deserves the
    forgiveness."""
    for gap, expect in ((-1e-8, True), (-1e-9, True), (-1e-11, True),
                        (-1e-13, False), (0.0, False), (1e-13, False),
                        (1e-11, False), (1e-9, False)):
        for x, y in _quad_family(gap):
            assert _shapes_overlap(x, y, _P, _P) is expect, (gap, x[0], y[0])


def test_w9d_tangency_and_abutment_stay_legal():
    """NON-DISCRIMINATOR (passes pre-fix).  The ONE-SIDED contract: an exactly
    tangent / abutting pair has a measure-zero intersection, so superposition of
    the form factors is still right and the geometry is LEGAL.  Checked on the
    solve path (not just the predicate) at 2000 random contact ANGLES, so the
    curved-pair reduction is exercised off the axes as well.  The radii are
    handed in as PYTHON floats deliberately: the predicate must return a Python
    ``bool`` (callers test it with ``is``), which a numpy-scalar semi-axis leaking
    a ``np.bool_`` would break -- found by this pin while writing it."""
    rng = np.random.default_rng(11)
    for _ in range(2000):
        r1, r2 = (float(v) for v in rng.uniform(0.03, 0.1, 2) * 1e-6)
        th = rng.uniform(0.0, 2.0 * np.pi)
        d = r1 + r2                                  # EXACT tangency
        a = ("disk", r1, r1, (0.3e-6, 0.3e-6))
        b = ("disk", r2, r2, (0.3e-6 + d * np.cos(th), 0.3e-6 + d * np.sin(th)))
        assert _shapes_overlap(a, b, _P, _P) is False, (r1, r2, th)
    # and through the public validator: exactly tangent disks + abutting boxes
    _validate_shapes("w9d", [
        {"shape": "disk", "eps": 6.25, "radius": 0.10e-6,
         "center": (0.15e-6, 0.30e-6)},
        {"shape": "disk", "eps": 6.25, "radius": 0.10e-6,
         "center": (0.35e-6, 0.30e-6)}], _P, _P)
    _validate_shapes("w9d", [
        {"shape": "rectangle", "eps": 6.25, "size": (0.2e-6, 0.3e-6),
         "center": (0.2e-6, 0.3e-6)},
        {"shape": "rectangle", "eps": 6.25, "size": (0.2e-6, 0.3e-6),
         "center": (0.4e-6, 0.3e-6)}], _P, _P)


def test_w9d_verdict_is_symmetric_in_pair_order():
    """NON-DISCRIMINATOR (passes pre-fix).  ``shapes[i]`` vs ``shapes[j]`` must
    not depend on which is which -- otherwise the same layout would validate or
    not depending on LIST ORDER.  The exact predicates erode BOTH shapes by
    ``tol/2`` precisely to keep this true (eroding only the second one is
    asymmetric for a mixed pair).  Measured 0 asymmetric verdicts in 20000
    random pairs."""
    rng = np.random.default_rng(4242)
    for _ in range(20000):
        a, b = _rand_shape(rng), _rand_shape(rng)
        assert _shapes_overlap(a, b, _P, _P) is _shapes_overlap(b, a, _P, _P)


def test_w9d_periodic_seam_is_still_seen():
    """NON-DISCRIMINATOR (passes pre-fix).  The wrap-aware MINIMAL periodic
    image survives the rewrite: a pair that is disjoint inside the cell but
    overlaps ACROSS the seam is still caught, for every pair kind."""
    for kind, geo in (("disk", {"radius": 0.12e-6}),
                      ("ellipse", {"semi_axes": (0.12e-6, 0.09e-6)}),
                      ("rectangle", {"size": (0.24e-6, 0.18e-6)})):
        shapes = [dict(shape=kind, eps=6.25, center=(0.02e-6, 0.30e-6), **geo),
                  dict(shape=kind, eps=6.25, center=(0.58e-6, 0.30e-6), **geo)]
        with pytest.raises(ValueError, match="OVERLAP"):
            _validate_shapes("w9d", shapes, _P, _P)
        # ... and the same pair moved off the seam is legal
        ok = [dict(shape=kind, eps=6.25, center=(0.16e-6, 0.30e-6), **geo),
              dict(shape=kind, eps=6.25, center=(0.44e-6, 0.30e-6), **geo)]
        _validate_shapes("w9d", ok, _P, _P)


def test_w9d_performance_envelope():
    """NON-DISCRIMINATOR (passes pre-fix).  The W8 envelope was 1024 shapes in
    81 ms (256 in 7.9 ms).  Exact algebra is FASTER than a 4096-direction scan
    per close pair: measured 51.8 ms for 1024 disks, 60.1 ms for 1024 ellipses
    and 54.4 ms for 1024 NEARLY-TOUCHING ellipses (where every neighbour reaches
    the predicate rather than being pre-filtered).  Gated at a generous 4 s --
    a CI box is not a benchmark, and the point of the pin is to catch an
    accidental O(K^2)-in-the-predicate regression, not to race."""
    for kind, geo in (("disk", lambda s: {"radius": 0.40 * s}),
                      ("ellipse", lambda s: {"semi_axes": (0.40 * s, 0.30 * s)}),
                      ("ellipse", lambda s: {"semi_axes": (0.4999 * s,
                                                           0.4999 * s)})):
        k = 32
        step = _P / k
        shapes = [dict(shape=kind, eps=6.0,
                       center=((i + 0.5) * step, (j + 0.5) * step),
                       **geo(step))
                  for i in range(k) for j in range(k)]
        assert len(shapes) == 1024
        t0 = time.perf_counter()
        _validate_shapes("w9d", shapes, _P, _P)
        dt = time.perf_counter() - t0
        assert dt < 4.0, (kind, dt)


# ===========================================================================
# W9-D2 -- the exactness itself
# ===========================================================================

def test_w9d_point_ellipse_distance_is_exact():
    """The one transcendental primitive, against a brute-force boundary scan of
    2e6 samples: the scan can only OVER-estimate the true distance, and it does
    so by <= 1.5e-09 relative -- i.e. the bracketed root agrees with the
    geometry to sampling precision.  Also pins the analytic special cases where
    the foot of the perpendicular is a VERTEX (a point on either axis)."""
    from lumenairy.elements.rcwa._core import _point_ellipse_distance
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 2.0 * np.pi, 2_000_001)
    ct, stt = np.cos(t), np.sin(t)
    for _ in range(12):
        p, q = rng.uniform(0.02, 0.3, 2) * 1e-6
        rad, th = rng.uniform(1.02, 4.0), rng.uniform(0, 2 * np.pi)
        u, v = rad * p * np.cos(th), rad * q * np.sin(th)
        d = _point_ellipse_distance(u, v, p, q)
        brute = float(np.min(np.hypot(u - p * ct, v - q * stt)))
        assert brute >= d * (1.0 - 1e-12), (p, q, u, v, d, brute)
        assert brute - d < 1e-8 * max(d, 1e-30), (p, q, u, v, d, brute)
    # vertices: a point on the x axis at u > p is (u - p) away, and the y twin
    for p, q in ((0.10e-6, 0.04e-6), (0.04e-6, 0.10e-6), (0.07e-6, 0.07e-6)):
        for f in (1.001, 1.5, 7.0):
            assert abs(_point_ellipse_distance(f * p, 0.0, p, q)
                       - (f - 1.0) * p) < 1e-14 * p
            assert abs(_point_ellipse_distance(0.0, f * q, p, q)
                       - (f - 1.0) * q) < 1e-14 * q


def test_w9d_exact_predicate_agrees_with_the_old_scan_at_the_shipped_window():
    """The rewrite is a DROP-IN at the shipped tolerance: 20000 random pairs
    across all six kind combinations, 0 disagreements with the pre-W9
    4096-direction scan (which is re-implemented in this file, so the two are
    genuinely independent)."""
    rng = np.random.default_rng(20260727)
    n_over = 0
    for _ in range(20000):
        a, b = _rand_shape(rng), _rand_shape(rng)
        new = _shapes_overlap(a, b, _P, _P)
        assert new is _old_scan(a, b, 1e-6), (a, b, new)
        n_over += bool(new)
    assert 0.05 < n_over / 20000 < 0.95, ("degenerate sample", n_over)


def test_w9d_the_old_scan_could_not_be_tightened():
    """WHY exactness was needed, measured.  Tighten the window and the old scan
    starts REJECTING LEGAL geometry: on 3000 exactly-tangent / gapped disk pairs
    at random contact angles (radii kept small enough that no periodic image can
    overlap, so every pair really is legal) it reports 406 false positives at
    ``tol_frac = 1e-8`` and 735 at 1e-10, because its separating-axis maximum is
    under-estimated by more than the window.  The exact predicate reports ZERO
    at every tolerance.  That is what makes the tolerance mean something."""
    rng = np.random.default_rng(4242)
    pairs = []
    for _ in range(3000):
        r1, r2 = rng.uniform(0.03, 0.1, 2) * 1e-6
        th = rng.uniform(0, 2 * np.pi)
        gap = float(rng.choice([0.0, 1e-13, 1e-11, 1e-9]))
        d = r1 + r2 + gap
        pairs.append((("disk", r1, r1, (0.3e-6, 0.3e-6)),
                      ("disk", r2, r2, (0.3e-6 + d * np.cos(th),
                                        0.3e-6 + d * np.sin(th)))))
    for tf, old_min in ((1e-8, 100), (1e-10, 100)):
        n_old = sum(_old_scan(a, b, tf) for a, b in pairs)
        n_new = sum(_tol_overlap(a, b, tf) for a, b in pairs)
        assert n_old > old_min, ("the old scan was expected to mis-fire", tf,
                                 n_old)
        assert n_new == 0, (tf, n_new)
    # at the SHIPPED window both are clean -- which is why W8 shipped safely
    assert sum(_old_scan(a, b, 1e-6) for a, b in pairs) == 0
    assert sum(_tol_overlap(a, b, 1e-6) for a, b in pairs) == 0


@pytest.mark.parametrize("depth", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
def test_w9d_detection_floor_reaches_1e_12_of_a_period(depth):
    """THE NEW FLOOR.  With the window set below the overlap depth, an overlap
    of ``depth`` PERIODS is reported for every pair kind -- measured down to
    1e-14 of a period (6e-21 m on this cell), 1e-12 pinned here with a 10x
    margin over the ``tol_frac = 1e-13`` window.  The pre-W9 predicate could not
    do this at ANY window (see
    :func:`test_w9d_the_old_scan_could_not_be_tightened`)."""
    for x, y in _quad_family(-depth * _P):
        assert _tol_overlap(x, y, 1e-13) is True, (depth, x[0], y[0])


def test_w9d_tangency_survives_the_tightest_window():
    """The other side of the floor: tightening the window must NOT start
    reporting tangency.  At exactly 0 separation -- and at gaps of 1e-12 /
    1e-10 / 1e-8 of a period -- every pair kind stays legal at
    ``tol_frac = 1e-13``, so the floor is one-sided all the way down."""
    for gap in (0.0, 1e-12 * _P, 1e-10 * _P, 1e-8 * _P):
        for x, y in _quad_family(gap):
            assert _tol_overlap(x, y, 1e-13) is False, (gap, x[0], y[0])


def test_w9d_curved_pairs_are_exact_off_the_axes():
    """The reduction must be exact for a CONTACT that is not on an axis, which is
    where the old scan's directional discretization lost the most.  Two ellipses
    are placed EXACTLY tangent at a known DIAGONAL point, by construction: scale
    A to the unit disk, pick the contact ``Q = (cos th, sin th)`` on that circle,
    find the point of B whose outward normal is ``-Q`` (for semi-axes
    ``(p, q)``: the boundary point at ``(cos s, sin s) ~ -(p cos th, q sin th)``),
    and translate B so the two points coincide -- a common tangent LINE, not the
    line-of-centres shortcut that only holds for circles.  Then push B out / in
    along ``Q`` by 3e-9 of the contact distance."""
    ax, ay = 0.11e-6, 0.05e-6
    bx, by = 0.06e-6, 0.09e-6
    a = ("ellipse", ax, ay, (0.30e-6, 0.30e-6))
    p, q = bx / ax, by / ay              # B's semi-axes in A-scaled coordinates
    for th in (0.3, 0.9, 1.7, 2.6, 4.1, 5.5):
        cth, sth = np.cos(th), np.sin(th)
        n = np.hypot(p * cth, q * sth)
        # the point of B (centred at the origin) whose outward normal is -Q
        pbx, pby = -p * p * cth / n, -q * q * sth / n
        cx0, cy0 = cth - pbx, sth - pby           # translate it onto Q
        for push, expect in ((-3e-9, True), (0.0, False), (3e-9, False)):
            cx, cy = cx0 + push * cth, cy0 + push * sth
            b = ("ellipse", bx, by,
                 (0.30e-6 + cx * ax, 0.30e-6 + cy * ay))
            assert _tol_overlap(a, b, 1e-11) is expect, (th, push)


def test_w9d_exactness_reaches_the_public_entry_points():
    """The floor is worth nothing if only the private predicate has it: a
    deliberately-constructed 1e-8-of-a-period overlap must still be REJECTED by
    ``add_layer(shapes=...)`` at the shipped window when it is deep enough, and
    the message must still name both shapes.  (At the shipped 1e-6 window a
    1e-8-of-a-period graze is deliberately FORGIVEN -- pinned here so the two
    layers of the contract cannot drift apart.)"""
    deep = [{"shape": "ellipse", "eps": 6.25, "semi_axes": (0.10e-6, 0.08e-6),
             "center": (0.20e-6, 0.30e-6)},
            {"shape": "disk", "eps": 6.25, "radius": 0.08e-6,
             "center": (0.20e-6 + 0.10e-6 + 0.08e-6 - 1e-5 * _P, 0.30e-6)}]
        # 1e-5 of a period deep: past the window, must raise
    st = RCWAStack(_P, period_y=_P, n_orders=3, n_orders_y=3)
    with pytest.raises(ValueError, match="shapes\\[0\\].*shapes\\[1\\].*OVERLAP"):
        st.add_layer(_D, shapes=deep, eps_background=1.0)
    graze = [dict(deep[0]),
             {"shape": "disk", "eps": 6.25, "radius": 0.08e-6,
              "center": (0.20e-6 + 0.10e-6 + 0.08e-6 - 1e-8 * _P, 0.30e-6)}]
    st2 = RCWAStack(_P, period_y=_P, n_orders=3, n_orders_y=3)
    st2.add_layer(_D, shapes=graze, eps_background=1.0)      # forgiven
    res = st2.set_source(_WL).solve()
    _o, R, T = (np.asarray(v) for v in res.efficiencies())
    for row in range(R.shape[0]):          # closure is PER incident polarization
        assert abs(float(R[row].sum() + T[row].sum()) - 1.0) < 1e-10, row
    # ... and the SAME graze is caught once the window is tightened
    a = ("ellipse", 0.10e-6, 0.08e-6, (0.20e-6, 0.30e-6))
    b = ("disk", 0.08e-6, 0.08e-6,
         (0.20e-6 + 0.10e-6 + 0.08e-6 - 1e-8 * _P, 0.30e-6))
    assert _tol_overlap(a, b, 1e-10) is True


def test_w9d_slack_is_a_named_constant_not_a_magic_number():
    """The blindness must be ONE named, documented number -- that is the whole
    point of removing the scan.  ``_OVERLAP_SLACK_FRAC`` is the default, the
    keyword overrides it, and the docstring says the predicates are exact."""
    from lumenairy.elements.rcwa import _core
    assert _core._OVERLAP_SLACK_FRAC == 1e-6
    assert "_OVERLAP_SLACK_FRAC" in _core.__all__
    doc = _shapes_overlap.__doc__
    for needle in ("EXACT", "tol_frac", "ONE-SIDED", "POINT-ELLIPSE",
                   "AXIS-ALIGNED"):
        assert needle in doc, needle
    a = ("disk", _R, _R, (0.15e-6, 0.3e-6))
    b = ("disk", _R, _R, (0.15e-6 + 2 * _R - 1e-8 * _P, 0.3e-6))
    assert _shapes_overlap(a, b, _P, _P) is False          # default: forgiven
    assert _tol_overlap(a, b, _core._OVERLAP_SLACK_FRAC) is False
    assert _tol_overlap(a, b, 1e-10) is True               # tightened: caught


def test_w9d_degenerate_and_extreme_geometry_is_handled():
    """The algebra must not blow up on the shapes the validator does accept:
    a very eccentric ellipse (100:1), a shape whose semi-axis is BELOW the
    erosion, a concentric pair, and a pair whose centres coincide."""
    tiny = 1e-7 * _P                     # below tol/2 = 3e-13 m
    a = ("ellipse", 0.2e-6, tiny, (0.3e-6, 0.3e-6))
    b = ("disk", 0.05e-6, 0.05e-6, (0.3e-6, 0.3e-6))
    assert _shapes_overlap(a, b, _P, _P) is False    # eroded away, not a crash
    # a 100:1 CROSS of two eccentric ellipses: they intersect for any y offset
    # that keeps the thin one crossing the wide one, and separate once the thin
    # one is moved clear in x (the wide one is only 5 nm tall).
    ecc = ("ellipse", 0.25e-6, 0.0025e-6, (0.3e-6, 0.3e-6))
    for dy in (0.0, 0.002e-6, 0.02e-6):
        other = ("ellipse", 0.0025e-6, 0.25e-6, (0.3e-6, 0.3e-6 + dy))
        assert _shapes_overlap(ecc, other, _P, _P) is True, dy
    for dx, expect in ((0.2e-6, True), (0.2529e-6, False), (0.28e-6, False)):
        other = ("ellipse", 0.0025e-6, 0.25e-6, (0.3e-6 + dx, 0.3e-6))
        assert _shapes_overlap(ecc, other, _P, _P) is expect, dx
    same = ("disk", 0.05e-6, 0.05e-6, (0.3e-6, 0.3e-6))
    assert _shapes_overlap(same, same, _P, _P) is True     # concentric
    nested = ("ellipse", 0.2e-6, 0.15e-6, (0.3e-6, 0.3e-6))
    assert _shapes_overlap(nested, same, _P, _P) is True   # one inside another
