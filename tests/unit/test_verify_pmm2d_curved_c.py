"""INDEPENDENT VERIFIER decision tests -- curved-cell Phase C of the pure
staggered 2-D PMM (``docs/audits/VERIFY_PMM2D_CURVED_C_2026_10_03.md``).

They close the gaps the verifier's mutation matrix found in the Phase C
gates (``validation/probe_pmm2d_curved/verify_c/v7_mutation_matrix.json``):
a merge that silently builds a WRONG map -- one layer's curves dropped, an
interior crossing of a curved edge put on the chord, the sinusoid's phase
lost, two different layouts of one outline accepted -- and a layer painted
in the wrong order or on the wrong background passed all 21 Phase C ids.
The oracle here reads nothing of the merge's own bookkeeping: it maps
interior points of every ``(u, v)`` cell through the map and asks the
SHAPES' analytic ``contains`` which material is there, and it samples every
material boundary of the painted grid through the map and asks which two
materials lie on either side.

Two further tests PIN open defects (``xfail(strict=True)``): they flip to
XPASS -- and fail, so the marker must be removed -- when the defect is
fixed with the edits given in the verify doc (V-D1 the rectangles-only
identity test misses round-off; V-D2 the rotated ellipse folds over most of
its documented angle range).

Fixture: square period 1.2 (lambda = 1 in the doc's probes), eps 4 / 2.25
features on vacuum unless stated.  Geometry only (no eigen-solve): each test
runs in well under a second per scenario.
"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import (Circle, Ellipse, FilletRect, Rect,
                                    SinusoidalWall, compile_shapes)
from lumenairy.elements.pmm import shapes2d as SH

_P = 1.2


def _paint(shapes, bg):
    def eps_at(x, y):
        out = np.full(np.shape(x), complex(bg))
        for sh in shapes:
            out[np.asarray(sh.contains(x, y), bool)] = complex(sh.eps)
        return out
    return eps_at


def _near_outline(shapes, x, y, tol):
    d = np.full(np.shape(x), np.inf)
    for sh in shapes:
        d = np.minimum(d, np.abs(np.asarray(sh.signed_distance(x, y))))
    return d <= tol


def _wrong_map(cmap, cells, layer_shapes, layer_bg):
    """(wrong interior points, wrong boundary points) summed over layers:
    the analytic material at the PHYSICAL image of 5 x 5 interior Gauss
    points of every cell must be the cell's eps (points within 1e-9 of an
    outline skipped), and at 24 points of every material boundary of the
    painted grid (periodic seams included) the analytic materials 1e-7 p
    either side along the image's normal must be the two cells' two eps."""
    U, V = cmap.u_bounds, cmap.v_bounds
    nx, ny = cmap.shape
    s5 = 0.5 + 0.5 * leggauss(5)[0]
    s = np.linspace(0.0, 1.0, 26)
    bad_in = bad_b = 0
    for cell, shapes, bg in zip(cells, layer_shapes, layer_bg):
        eps_at = _paint(shapes, bg)
        for i in range(nx):
            for j in range(ny):
                X, Y = cmap.geom(i, j, U[i] + s5 * (U[i + 1] - U[i]),
                                 V[j] + s5 * (V[j + 1] - V[j]))[:2]
                far = ~_near_outline(shapes, X, Y, 1e-9)
                bad_in += int(np.sum((np.abs(eps_at(X, Y) - cell[i, j])
                                      > 1e-12) & far))
                for side, (a, b) in (("r", ((i + 1) % nx, j)),
                                     ("t", (i, (j + 1) % ny))):
                    if cell[i, j] == cell[a, b]:
                        continue
                    if side == "r":
                        Uq = np.full(s.size, U[i + 1])
                        Vq = V[j] + s * (V[j + 1] - V[j])
                    else:
                        Uq = U[i] + s * (U[i + 1] - U[i])
                        Vq = np.full(s.size, V[j + 1])
                    X, Y = cmap.geom_points(i, j, Uq, Vq)[:2]
                    P = np.stack([X, Y], 1)
                    tg = P[2:] - P[:-2]
                    tg /= np.hypot(tg[:, 0], tg[:, 1])[:, None]
                    nrm = np.stack([tg[:, 1], -tg[:, 0]], 1) * 1e-7 * _P
                    pa, pb = P[1:-1] + nrm, P[1:-1] - nrm
                    ea = eps_at(np.mod(pa[:, 0], cmap.period_x),
                                np.mod(pa[:, 1], cmap.period_y))
                    eb = eps_at(np.mod(pb[:, 0], cmap.period_x),
                                np.mod(pb[:, 1], cmap.period_y))
                    want = {complex(cell[i, j]), complex(cell[a, b])}
                    bad_b += sum(1 for x, y in zip(ea, eb)
                                 if {complex(x), complex(y)} != want)
    return bad_in, bad_b


def test_vc1_painting_order_and_background_eps_are_honoured():
    """Within a layer a LATER shape covers an earlier one, onto
    ``background_eps``.  The compile_shapes docstring states the hole-in-a-
    slab cells; a circle on a 2.25 background leaves 2.25 outside.
    Mutation matrix: painting reversed and background_eps ignored both
    passed all 21 Phase C ids (every Phase C test paints on vacuum, and the
    route-equality tests compare the merge with itself)."""
    slab = Rect(0.6, 0.6, _P, _P, 12.1)
    hole = Circle(0.6, 0.6, 0.3, 1.0)
    eps, _x, _y, _cm = compile_shapes(_P, _P, [slab, hole], 1.0)
    want = np.full((3, 3), 12.1 + 0j)
    want[1, 1] = 1.0
    assert np.array_equal(eps, want)
    eps2, _x, _y, _cm = compile_shapes(_P, _P, [Circle(0.6, 0.6, 0.3, 4.0)],
                                       2.25)
    want2 = np.full((3, 3), 2.25 + 0j)
    want2[1, 1] = 4.0
    assert np.array_equal(eps2, want2)


_SCENARIOS = {
    # two layers whose curves are CUT by the other layer's walls (interior
    # crossings of curved edges; the annulus device)
    "annulus": (_P, [([Circle(0.6, 0.6, 0.25, 4.0)], 1.0),
                     ([Circle(0.6, 0.6, 0.45, 2.25)], 1.0)]),
    # a PHASED sinusoid (axis y, two waves) beside a circle in the other
    # layer, on a 2.25 background
    "phased_sine_circle": (_P, [([SinusoidalWall("y", 0.15, 0.05, 2, 0.7,
                                                 eps=4.0)], 2.25),
                                ([Circle(0.6, 0.62, 0.3, 4.0)], 1.0)]),
    # a circle whose wall cuts a fillet's arc row in the other layer
    "circle_cuts_fillet_row": (2.0, [([Circle(0.6, 0.6, 0.0849, 4.0)], 1.0),
                                     ([FilletRect(1.4, 1.0, 0.6, 0.8, 0.1,
                                                  2.25)], 1.0)]),
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_vc2_merged_map_is_exact_against_the_analytic_outlines(name):
    """The merged map of a multi-layer stack reproduces EVERY layer's
    analytic outline: no interior point of any cell maps into the wrong
    material, and every material boundary of the painted grid sits on an
    outline (the materials either side are the right two).  Measured on
    this tree 2026-10-03 (``v2_merge_win.json``): 0 wrong points of 3200
    boundary samples and of every interior sample.  Mutation matrix (all
    passed the 21 Phase C ids): layer >= 2 curves dropped -> its disk
    becomes a polygon (wrong interior points); an interior crossing of a
    curved edge placed on the chord -> the map cannot be built; the
    sinusoid's phase dropped from the map -> the wall sits 0.05 sin(0.7)
    off its outline."""
    period, layers = _SCENARIOS[name]
    lab = [(f"layer {k + 1}", sh, bg) for k, (sh, bg) in enumerate(layers)]
    _U, _V, cmap, cells, ident = SH._merge(period, period, lab)[:5]
    assert not ident
    bad_in, bad_b = _wrong_map(cmap, cells, [sh for sh, _ in layers],
                               [bg for _, bg in layers])
    assert bad_in == 0 and bad_b == 0, (bad_in, bad_b)


def test_vc3_one_outline_in_two_layouts_is_refused_not_merged():
    """The SAME circle in two layers with two layouts (3 x 3 and the 5 x 5
    ``core=0.5``) claims one grid vertex at two physical points 1.05e-2
    apart (uniform-in-angle arc cut vs the core layout's own vertex); the
    merge must refuse it, naming both.  (An honest over-refusal -- the
    outline is identical -- but the claim tolerance is what stands between
    a merge and a silently mixed map.)  Mutation matrix: the claim
    tolerance loosened to 1e-2 p passed all 21 Phase C ids."""
    with pytest.raises(ValueError, match=r"DIFFERENT physical positions"):
        SH._merge(_P, _P, [("layer 1", [Circle(0.6, 0.6, 0.36, 4.0)], 1.0),
                           ("layer 2", [Circle(0.6, 0.6, 0.36, 2.0,
                                               core=0.5)], 1.0)])


@pytest.mark.xfail(strict=True, reason="V-D1: a rectangles-only merge whose "
                   "walls coincide only to round-off is not recognised as "
                   "the identity map (mapped solver; tensors refused)")
def test_vc4_rectangles_only_merge_is_the_identity_to_round_off():
    """Rectangles need no map, and the stack then runs the UNMAPPED solver
    (and accepts tensors).  Two layers of rectangles whose shared wall is
    computed two ways (0.6 - 0.15 vs 0.3 + 0.15, one ulp apart) and whose
    straight edges are cut by the other layer's walls: the vertex claims
    land 1 ulp off the merged walls and ``_merge`` reports a NON-identity
    map -- measured on 7.5 % of random two-rectangle layouts
    (``v3_struct_win.json``; the stack then takes the quadrature path and
    refuses a tensor as 'CURVED').  The fix (verify doc V-D1) snaps a claim
    within 1e-13 p of its grid vertex onto it."""
    lay = [("layer 1", [Rect(0.45, 0.6, 0.5, 0.3, 4.0),
                        Rect(0.95, 0.3, 0.2, 0.3, 2.25)], 1.0),
           ("layer 2", [Rect(0.6, 0.6, 1.2, 0.4, 3.0)], 1.5)]
    assert SH._merge(_P, _P, lay)[4]


@pytest.mark.xfail(strict=True, reason="V-D2: the rotated Ellipse's layout "
                   "(corners at the PARAMETRIC 45-degree points) folds for "
                   "aspect 2 at 20 deg, aspect 5 at 10 deg")
def test_vc5_rotated_ellipse_lays_out_over_its_documented_range():
    """``Ellipse(..., angle=)`` documents ``|angle| < 45 deg``.  Measured
    (``v5b_ellipse_layout_win.json``): the shipped layout FOLDS for aspect
    1.5 at 30 deg, 2 at 20 deg, 3 at 20 deg, 5 at 10 deg (and 1.05 at 40
    deg); corners where the outward NORMAL points at 45 / 135 / 225 / 315
    degrees (the circle's 45-degree points' analogue, verify doc V-D2) lay
    out every case exactly up to 44.9 deg and aspect 5."""
    for a, b, ang in ((0.4, 0.2, 20.0), (0.4, 0.08, 10.0),
                      (0.35, 0.2, 44.9)):
        e = Ellipse(0.6, 0.6, a, b, 4.0, angle=np.deg2rad(ang))
        cell, _x, _y, cm = compile_shapes(_P, _P, [e], 1.0)
        assert _wrong_map(cm, [cell], [[e]], [1.0]) == (0, 0)
