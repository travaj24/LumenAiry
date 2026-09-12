"""WP-A13 (audit 2026-09-11), finding **G9** -- ``eps_cell`` means two
INCOMPATIBLE things across the two 2-D PMM families, and the mistake was an
unguarded ~1000x cost cliff.

* the HYBRID family (``pmm_efficiency_2d_cell`` / ``pmm_jones_2d`` /
  ``PMM2DStackHybrid.add_layer``) takes a **PIXEL** grid: redundant rows and
  columns are merged away by ``_cell_to_walls_tile``, so a 12x12 array and the
  3x3 that describes the same half-fill pillar cost the same, and a cost guard
  (``max_nodal_dof``) already existed;
* the STAGGERED family (``pmm_efficiency_2d_staggered`` /
  ``pmm_jones_2d_staggered`` / ``PMM2DStackPure.add_layer``) takes a
  **SEGMENT** grid: every row and column IS an element, the generalized pencil
  is ``2 * Nx*(M-1) * Ny*(M-1)``, and there was NO cost guard at all.

MEASURED on that exact 12x12 array: ``pmm_efficiency_2d_staggered(degree=6)``
498 s CPU / 8.4 GB (a 7200x7200 QZ pencil where an 800x800 one suffices) and
``PMM2DStackPure(n_modes=5).add_layer`` 1096 s / 3.9 GB, against **0.22 s** and
< 1 GB for the same array through ``pmm_efficiency_2d_cell`` -- and neither
staggered call raised, warned or printed anything.

Every assertion here is on the GUARD, never on a solve, so this file is fast:
the whole point is that the expensive call is stopped before it starts.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStackPure,
    pmm_efficiency_2d_staggered,
)
from lumenairy.elements.pmm.twod_staggered import (
    _MAX_STAG_PENCIL_DOF,
    _stag_merged_segments,
)

_WL, _P, _DEP = 1.0e-6, 0.9e-6, 0.3e-6


def _pillar_12():
    """The audit's fixture: a centred half-fill pillar drawn on 12 segments per
    axis.  It has 3 DISTINCT strips, and its walls (indices 3 and 9, i.e. 1/4
    and 3/4) sit on the uniform 4-segment lattice -- so 4 segments, not 3, is
    the smallest grid that expresses it exactly."""
    c = np.full((12, 12), 1.0 + 0j)
    c[3:9, 3:9] = 12.25
    return c


def _pillar_3():
    return np.array([[1.0, 1.0, 1.0], [1.0, 12.25, 1.0], [1.0, 1.0, 1.0]],
                    dtype=complex)


def test_g9_merged_segment_count_is_the_hybrid_merge_rule():
    """The merged count is the same quantity ``twod._cell_to_walls_tile``
    derives -- a boundary survives only where the adjacent row/column actually
    differs -- so the message can name the grid the user should have passed."""
    from lumenairy.elements.pmm.twod import _cell_to_walls_tile
    for cell in (_pillar_12(), _pillar_3()):
        xw, yw, _tile = _cell_to_walls_tile(cell, _P, _P, "t")
        assert _stag_merged_segments(cell) == (len(xw) + 1, len(yw) + 1)
    assert _stag_merged_segments(_pillar_12()) == (3, 3)
    # several arrays merge JOINTLY (eps + mu): a boundary survives where EITHER
    # changes
    a = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    b = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=complex)
    assert _stag_merged_segments(a) == (1, 1)
    assert _stag_merged_segments(a, b) == (2, 1)


@pytest.mark.parametrize("call", ["entry", "stack"])
def test_g9_an_unaffordable_segment_grid_raises_before_solving(call):
    """The absolute budget.  ``max_pencil_dof=1`` forces the raise on a grid
    that is otherwise legal, so the test never pays the solve; the message must
    name the pencil, the PIXEL-vs-SEGMENT distinction and the reduced grid."""
    cell = _pillar_12()
    with pytest.raises(ValueError) as ei:
        if call == "entry":
            pmm_efficiency_2d_staggered(_P, _P, cell, 1.45, 1.0, _DEP, _WL,
                                        degree=6, n_orders=3, max_pencil_dof=1)
        else:
            PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                           n_modes=5, n_orders=3).add_layer(
                               _DEP, eps_cell=cell, max_pencil_dof=1)
    msg = str(ei.value)
    # the pencil the audit measured at 498 s / 8.4 GB (M=6) and 1096 s / 3.9 GB
    # (M=5), exactly 2 * (12*(M-1))**2
    assert ("7200x7200" in msg) if call == "entry" else ("4608x4608" in msg)
    assert "SEGMENT grid" in msg and "PIXEL grid" in msg
    assert "only 3x3 DISTINCT strips" in msg
    assert "uniform 4x4 lattice" in msg
    assert "max_pencil_dof" in msg


def _stack_warnings(cells, *, n_modes=5, layer_grids="shared", **kw):
    """Every SEGMENT-grid warning a stack produces, from ``add_layer`` AND from
    the joint check ``solve`` runs first -- WITHOUT solving, so nothing
    expensive runs (VERIFY-A13 V5 moved the shared path's advice to solve
    time; ``_warn_stag_shared_redundancy`` is the first thing ``solve`` calls
    and runs no eigensolve)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=n_modes, n_orders=3,
                            layer_grids=layer_grids)
        for c in cells:
            st.add_layer(_DEP, eps_cell=c, **kw)
        st._warn_stag_shared_redundancy()
        return [str(x.message) for x in w if "SEGMENT grid" in str(x.message)]


def test_g9_a_redundant_patterned_grid_warns_with_the_merged_count():
    """At the DEFAULT budget the audit's call is affordable-ish and therefore
    proceeds -- but it must say that 12 segments/axis is 729x the QZ work of
    the 4 the geometry needs.  Splitting a region into more segments IS a legal
    h-refinement, which is why this warns rather than refusing.

    A one-patterned-layer stack is where the joint rule (VERIFY-A13 V5) reduces
    EXACTLY to the per-layer one, so the audit's case reads the same number it
    always did -- it is now delivered at the top of ``solve``, before the QZ it
    is warning about, rather than at ``add_layer``."""
    msgs = _stack_warnings([_pillar_12()])
    assert len(msgs) == 1, msgs
    assert "12x12 segments and only 3x3 DISTINCT strips" in msgs[0]
    # the SUGGESTION is the smallest uniform lattice that still contains the
    # walls (indices 3 and 9 of 12 -> 1/4 and 3/4), which is 4 segments, NOT
    # the 3 DISTINCT strips: a 3-segment lattice puts its walls at 1/3 and 2/3
    # and would silently change the duty cycle.
    assert "uniform 4x4 lattice" in msgs[0]
    assert "4608x4608" in msgs[0] and "512x512" in msgs[0]
    assert "729x less QZ time" in msgs[0]


def test_g9_a_minimal_or_uniform_grid_is_silent():
    """The guard must be a signal, not noise.

    Two exemptions, both measured decisions rather than taste:

    * a grid with no two adjacent identical rows/columns is already minimal;
    * a grid that merges to 1x1 is UNIFORM -- tiling a uniform axis into equal
      segments is the DOCUMENTED way to satisfy this family's ``Nx == Ny``
      contract, and h-refining a uniform region is a real accuracy lever here
      (``PMM2DStackPure.add_layer(grid=)``), so it is a deliberate choice
      rather than the PIXEL-grid mistake.  The absolute budget still guards it.
    """
    uni = np.full((3, 3), 2.25 + 0j)
    corner = np.full((6, 6), 1.0 + 0j)
    corner[0, 0] = 12.25          # 2 distinct strips/axis, but wall at 1/6
    for cell in (_pillar_3(), uni, corner):
        assert not _stack_warnings([cell]), cell


def test_g9_the_suggestion_is_a_lattice_that_actually_holds_the_walls():
    """The distinct-strip count is NOT a safe suggestion, and this is the
    test that keeps it from becoming one.

    Two ways it would be wrong advice, both pinned here:

    * the walls must survive.  The audit's 12x12 half-fill pillar has 3
      distinct strips but its walls sit at indices 3 and 9, i.e. at 1/4 and
      3/4; a 3-segment uniform lattice puts them at 1/3 and 2/3, so
      "re-express it on a 3x3 grid" would silently change the DUTY CYCLE --
      the same failure mode as the sibling ``period_x`` finding.  The
      reducible factor is ``gcd(N, every wall index)``, giving 4.
    * the grid must stay SQUARE.  A 1-D stripe merges per axis to ``(2, 1)``,
      which the ``Nx == Ny`` guard rejects outright.

    So the suggestion is the smallest SQUARE UNIFORM lattice containing every
    wall on EITHER axis, and a cell already on it must be silent."""
    from lumenairy.elements.pmm.twod_staggered import (
        _stag_minimal_uniform_segments,
    )
    pillar = _pillar_12()
    assert _stag_merged_segments(pillar) == (3, 3)
    assert _stag_minimal_uniform_segments(pillar) == 4        # not 3
    # a 3x3 centred pillar (walls at 1/3, 2/3) is already minimal
    p3 = _pillar_3()
    assert _stag_merged_segments(p3) == (3, 3)
    assert _stag_minimal_uniform_segments(p3) == 3
    # a stripe: 2 distinct strips per axis, but its wall is at 1/3 on a 3-grid
    stripe3 = np.array([[12.25] * 3, [1.0] * 3, [1.0] * 3], dtype=complex)
    assert _stag_merged_segments(stripe3) == (2, 1)
    assert _stag_minimal_uniform_segments(stripe3) == 3       # not 2
    stripe2 = np.array([[12.25, 12.25], [1.0, 1.0]], dtype=complex)
    assert _stag_merged_segments(stripe2) == (2, 1)
    assert _stag_minimal_uniform_segments(stripe2) == 2
    for cell in (p3, stripe3, stripe2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                           n_modes=5, n_orders=3).add_layer(_DEP,
                                                            eps_cell=cell)
        assert not [x for x in w if "SEGMENT grid" in str(x.message)], cell
    # ... and the square guard really does reject a per-axis merge
    with pytest.raises(ValueError, match="must be SQUARE"):
        PMM2DStackPure(_P, _P, n_modes=5, n_orders=3).add_layer(
            _DEP, eps_cell=stripe2[:, :1])


def test_g9_the_default_budget_admits_every_grid_the_suite_uses():
    """The absolute cap is calibrated on the RESOURCE cliff, not on taste: the
    largest legitimate grid in the shipped staggered suite is 8 segments/axis
    at M = 8 (pencil 6272) and the documented M = 10 / 3-segment solve is 1458.
    Both must be admissible; the projected footprint at the cap is ~28 GB."""
    for N, M in ((2, 8), (3, 10), (6, 8), (8, 8), (2, 30)):
        assert 2 * (N * (M - 1)) ** 2 <= _MAX_STAG_PENCIL_DOF, (N, M)
    assert 12 * _MAX_STAG_PENCIL_DOF ** 2 * 16 / 2 ** 30 > 20.0


def test_g9_the_hybrid_docstrings_cross_reference_the_two_meanings():
    """CONVENTIONS sec 11 cross-references ``fff_nv`` between the two Jones
    engines; the same has to hold for ``eps_cell`` between the two 2-D PMM
    families, or the next reader repeats the 1000x mistake."""
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure as _SP
    from lumenairy.elements.pmm.twod import _cell_to_walls_tile
    assert "SEGMENT grid" in _cell_to_walls_tile.__doc__
    assert "PIXEL grid" in _cell_to_walls_tile.__doc__
    assert "PIXEL grid" in pmm_efficiency_2d_cell.__doc__
    assert "SEGMENT" in _SP.add_layer.__doc__
    assert "PIXEL" in _SP.add_layer.__doc__
    assert "SEGMENT grid, not a PIXEL grid" in \
        pmm_efficiency_2d_staggered.__doc__


# --------------------------------------------------------------------------- #
# VERIFY-A13 V5 -- the SHARED grid's redundancy advice is a STACK property
# --------------------------------------------------------------------------- #

def _tiled(n, lo, hi, reps):
    """An ``n``-segment pillar tiled ``reps`` times per axis: an
    ``n*reps``-segment cell whose minimal uniform lattice is ``n``."""
    c = np.full((n, n), 1.0 + 0j)
    c[lo:hi, lo:hi] = 12.25
    return np.kron(c, np.ones((reps, reps)))


def test_v5_the_shared_grids_advice_is_the_joint_minimum_not_the_per_layer_one():
    """``layer_grids='shared'`` makes the lattice a property of the STACK, so
    per-layer advice is UNFOLLOWABLE and must not be given.

    MEASURED before this rule: a 2x2 pillar and a 3x3 pillar tiled onto one
    shared 6x6 lattice each drew "expressible on the uniform 2x2 / 3x3
    lattice", and FOLLOWING either one made the other layer's ``add_layer``
    raise ``all patterned layers must share ONE common (Nx, Ny) grid``.  The
    grid the caller CAN pass is the smallest uniform lattice holding EVERY
    layer's walls -- the lcm of the per-layer minima, ``lcm(2, 3) = 6``, which
    is the grid they already passed.

    Exact counts, not tolerances: the pair on 6x6 is SILENT, and the identical
    pair tiled onto 12x12 warns ONCE naming 6.  ``_stag_minimal_uniform_
    segments`` over the joint wall set computes that lcm directly, because
    ``N / gcd(g_A, g_B) = lcm(N/g_A, N/g_B)`` with ``g_L = gcd(N, walls of L)``.
    """
    a6, b6 = _tiled(2, 0, 1, 3), _tiled(3, 1, 2, 2)         # both 6x6
    assert _stag_merged_segments(a6, b6)                     # joint merge runs
    assert not _stack_warnings([a6, b6], n_modes=3)

    a12, b12 = _tiled(2, 0, 1, 6), _tiled(3, 1, 2, 4)        # both 12x12
    msgs = _stack_warnings([a12, b12], n_modes=3)
    assert len(msgs) == 1, msgs
    assert "uniform 6x6 lattice" in msgs[0], msgs[0]
    # ... and it does NOT suggest either layer's own minimum
    assert "uniform 2x2 lattice" not in msgs[0]
    assert "uniform 3x3 lattice" not in msgs[0]


def test_v5_per_layer_grids_keep_the_per_layer_advice():
    """With ``layer_grids='per-layer'`` each layer really does own its lattice,
    so the per-layer suggestion IS followable and both layers must still get
    it -- the control that shows the change above is scoped to the union-grid
    contract and not a general weakening of the guard."""
    a6, b6 = _tiled(2, 0, 1, 3), _tiled(3, 1, 2, 2)
    msgs = _stack_warnings([a6, b6], n_modes=3, layer_grids="per-layer")
    assert len(msgs) == 2, msgs
    assert any("uniform 2x2 lattice" in m for m in msgs), msgs
    assert any("uniform 3x3 lattice" in m for m in msgs), msgs


def test_v5_max_pencil_dof_acknowledges_the_shared_stack_too():
    """``max_pencil_dof=`` is the documented acknowledgement of a deliberate
    h-refinement, and it has to reach the joint check as well -- otherwise a
    caller who acknowledged at ``add_layer`` would be warned again at solve."""
    a6 = _tiled(2, 0, 1, 3)
    assert _stack_warnings([a6], n_modes=3)                  # warns by default
    assert not _stack_warnings([a6], n_modes=3, max_pencil_dof=10 ** 7)


def test_v5_the_absolute_refusal_is_unchanged_and_still_at_add_layer():
    """Only the ADVICE moved.  The RAISE is an absolute cost bound, always
    actionable via ``max_pencil_dof=``, and must still fire where the geometry
    is accepted -- before any solve, on the shared path as on the per-layer
    one."""
    for grids in ("shared", "per-layer"):
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=5, n_orders=3, layer_grids=grids)
        with pytest.raises(ValueError, match=r"max_pencil_dof=1\b"):
            st.add_layer(_DEP, eps_cell=_pillar_12(), max_pencil_dof=1)
