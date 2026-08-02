"""Niche D4 -- DGRATING import + a DOE entry in the traced carrier chain.

Roadmap ``ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27`` P2, "Let the full
design be expressed as ONE object".

Two halves, both exercised here against SELF-CONTAINED synthetic fixtures (CI
has no ``.zmx``; the design-121 acceptance runs locally and is reported, never
pinned here):

1. **Import.**  ``DGRATING`` surfaces used to fall into the unknown-SURFTYPE
   branch: the geometry imported fine but PARM 1 (lines/um) and PARM 2 (the
   design diffraction order) were DROPPED, so the prescription the wave chain
   saw had never contained the DOE.  They now round-trip into
   ``rx['diffractives']`` -- period, order, grating azimuth (with any
   enclosing ``COORDBRK`` z-roll folded in), and the axial gaps to the
   neighbouring POWERED elements.  The geometry import is unchanged: still a
   flat/conic surface, still no aspheric coefficients, still nothing on the
   lens-only ``'surfaces'`` list.

2. **Chain bookkeeping.**  ``propagate_traced_carrier_chain``'s ``groups``
   list accepts a DOE entry, so the DOE's gaps stop being a manual fold.  The
   binding equivalence -- "a chain whose groups list includes a DOE entry
   reproduces the manual hand-split + hand-folded-gap arrangement" -- is
   pinned three ways here: gap_after <-> next gap_before is BITWISE, an
   undeflected DOE reproduces the hand-folded gap, and a deflected order
   reproduces the two-chain hand split that design 121's own fan runner does
   by hand.  The discriminator test shows what the manual FOLD (as opposed to
   the manual SPLIT) gets wrong, which is the failure this feature removes.

The DOE's per-order action is checked against inline oracles only: the
grating equation in direction cosines, an exact meridional ray trace through
the following group, and the grating's own one-period translation symmetry.

Two more sections pin what an adversarial re-measure caught in the first
revision of this work.  ``TestSplitLegPathDependence`` pins the REAL envelope
of "a split carrier leg is not inert" (the co-moving magnification telescopes
exactly except where the split plane lands in the near-focus bridge zone) --
the shipped justification for deferring the DOE's transport once cited that
mechanism on a leg that does not have it.  And the exit-power tests pin that
a DOE's order ``amplitude`` reaches ``power_exit`` from a TRAILING screen as
well as from one between groups, so the multi orchestrator's readout-clip
guard cannot fire on bookkeeping.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import propagate_traced_carrier_chain as chain
from lumenairy.io.prescriptions_zemax import load_zemax_zmx
from lumenairy.propagators.carrier import (
    _chain_chief_ray_at_target,
    _multi_chain_exit_power,
    _normalise_doe_entry,
    carrier_referenced_envelope,
    propagate_carrier_referenced,
)

LAM = 1.31e-6


# ==========================================================================
# .zmx fixtures -- minimal in-memory files (same style as test_audit_w5)
# ==========================================================================

_HEADER = [
    'VERS 210000 0 123 0 0',
    'MODE SEQ',
    'NAME d4_dgrating_test',
    'UNIT MM X W X CM MR CPMM',
    'ENPD 10.0',
    'WAVM 1 1.310000 1.0',
    'PWAV 1',
]


def _load(lines):
    fd, path = tempfile.mkstemp(suffix='.zmx', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            rx = load_zemax_zmx(path)
    finally:
        os.unlink(path)
    return rx, [str(w.message) for w in wlist]


def _crossed_doe_zmx():
    """The design-121 LAYOUT in miniature, and nothing else about it.

    Singlet -- 51.539 mm -- flat air-air STOP dummy -- DGRATING(x) --
    COORDBRK(+90 roll) -- DGRATING(y) -- COORDBRK(-90) -- flat air-air dummy
    -- 7 mm -- singlet.  Everything the gap bookkeeping has to survive is in
    there: a dummy plane before the DOE, a coordinate-break pair around it,
    and a dummy plane after it."""
    return _HEADER + [
        'SURF 0',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
        # --- first singlet -------------------------------------------------
        'SURF 1',
        '  TYPE STANDARD', '  CURV 0.01 0 0 0 0 ""', '  DISZ 3.0',
        '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0',
        'SURF 2',
        '  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""',
        '  DISZ 51.5393280925041', '  DIAM 6.0',
        # --- air-air dummy carrying the stop (collapses into the gap) ------
        'SURF 3',
        '  COMM truncate here', '  STOP',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 6.0',
        # --- the crossed DOE pair ------------------------------------------
        'SURF 4',
        '  COMM Diffractive', '  TYPE DGRATING', '  CURV 0.0',
        '  PARM 1 0.0087899999999999992', '  PARM 2 -4',
        '  DISZ 0.0', '  DIAM 6.5',
        'SURF 5',
        '  TYPE COORDBRK', '  CURV 0.0',
        '  PARM 1 0', '  PARM 2 0', '  PARM 3 0', '  PARM 4 0',
        '  PARM 5 90', '  PARM 6 0', '  DISZ 0.0', '  DIAM 0.0',
        'SURF 6',
        '  TYPE DGRATING', '  CURV 0.0',
        '  PARM 1 -0.0087899999999999992', '  PARM 2 -2',
        '  DISZ 0.0', '  DIAM 6.5',
        'SURF 7',
        '  TYPE COORDBRK', '  CURV 0.0',
        '  PARM 1 0', '  PARM 2 0', '  PARM 3 0', '  PARM 4 0',
        '  PARM 5 -90', '  PARM 6 0', '  DISZ 0.0', '  DIAM 0.0',
        # --- air-air dummy after the DOE (also collapses) ------------------
        'SURF 8',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 7.0', '  DIAM 6.5',
        # --- second singlet ------------------------------------------------
        'SURF 9',
        '  TYPE STANDARD', '  CURV 0.008 0 0 0 0 ""', '  DISZ 3.0',
        '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0',
        'SURF 10',
        '  TYPE STANDARD', '  CURV -0.008 0 0 0 0 ""', '  DISZ 40.0',
        '  DIAM 6.0',
        'SURF 11',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 3.0',
        'BLNK',
    ]


def _plain_doublet_zmx():
    """The SAME file with the whole DOE island removed -- the control."""
    return _HEADER + [
        'SURF 0',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
        'SURF 1',
        '  TYPE EVENASPH', '  CURV 0.01 0 0 0 0 ""', '  CONI -0.5',
        '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0',
        '  PARM 2 1.5e-05', '  DIAM 6.0',
        'SURF 2',
        '  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""', '  DISZ 58.5393280925041',
        '  DIAM 6.0',
        'SURF 3',
        '  TYPE STANDARD', '  CURV 0.008 0 0 0 0 ""', '  DISZ 3.0',
        '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0',
        'SURF 4',
        '  TYPE STANDARD', '  CURV -0.008 0 0 0 0 ""', '  DISZ 40.0',
        '  DIAM 6.0',
        'SURF 5',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 3.0',
        'BLNK',
    ]


SEP_T1, SEP_T2, SEP_T3 = 20.0, 10.0, 7.0     # mm, lens -> DOE -> DOE -> lens


def _separated_doe_zmx():
    """A pair of DGRATINGs with a REAL distance between them.

    ``lensA -- 20 mm -- DGRATING-A -- 10 mm -- DGRATING-B -- 7 mm -- lensB``.
    Design 121's crossed pair sits at ``DISZ 0.0``, and so does
    :func:`_crossed_doe_zmx`, so neither can tell whether the inter-DOE leg
    is recorded ONCE or TWICE.  This one can: with the leg double-counted the
    documented drop-in transports 47 mm instead of 37."""
    return _HEADER + [
        'SURF 0',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
        'SURF 1',
        '  TYPE STANDARD', '  CURV 0.01 0 0 0 0 ""', '  DISZ 3.0',
        '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0',
        'SURF 2',
        '  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""', f'  DISZ {SEP_T1}',
        '  DIAM 6.0',
        'SURF 3',
        '  COMM DOE-A', '  TYPE DGRATING', '  CURV 0.0',
        '  PARM 1 0.00879', '  PARM 2 -4', f'  DISZ {SEP_T2}', '  DIAM 6.5',
        'SURF 4',
        '  COMM DOE-B', '  TYPE DGRATING', '  CURV 0.0',
        '  PARM 1 0.00879', '  PARM 2 -2', f'  DISZ {SEP_T3}', '  DIAM 6.5',
        'SURF 5',
        '  TYPE STANDARD', '  CURV 0.008 0 0 0 0 ""', '  DISZ 3.0',
        '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0',
        'SURF 6',
        '  TYPE STANDARD', '  CURV -0.008 0 0 0 0 ""', '  DISZ 40.0',
        '  DIAM 6.0',
        'SURF 7',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 3.0',
        'BLNK',
    ]


def _doe_outside_glass_zmx(where, lines_per_um='0.00879'):
    """A DGRATING that does NOT sit between two glass surfaces.

    ``where='first'``: collimated -> DGRATING -> 12 mm -> singlet (a fan-out
    behind a collimator).  ``where='last'``: singlet -> 12 mm -> DGRATING ->
    9 mm -> dummy -> 5 mm -> image (a fan-out at the output).  Both are
    ordinary DOE layouts, and both used to come back with an EMPTY
    ``'diffractives'`` list and no warning, because the lens-window
    auto-detect clipped to the glass span and an air-to-air DGRATING is not
    in it."""
    doe = ['  COMM DOE', '  TYPE DGRATING', '  CURV 0.0']
    if lines_per_um is not None:
        doe.append(f'  PARM 1 {lines_per_um}')
    doe.append('  PARM 2 -4')
    obj = ['SURF 0', '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY',
           '  DIAM 5.0']
    img = ['  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 3.0',
           'BLNK']
    if where == 'first':
        return (_HEADER + obj
                + ['SURF 1'] + doe + ['  DISZ 12.0', '  DIAM 6.5']
                + ['SURF 2', '  TYPE STANDARD', '  CURV 0.01 0 0 0 0 ""',
                   '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0']
                + ['SURF 3', '  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""',
                   '  DISZ 40.0', '  DIAM 6.0']
                + ['SURF 4'] + img)
    return (_HEADER + obj
            + ['SURF 1', '  TYPE STANDARD', '  CURV 0.01 0 0 0 0 ""',
               '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 6.0']
            + ['SURF 2', '  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""',
               '  DISZ 12.0', '  DIAM 6.0']
            + ['SURF 3'] + doe + ['  DISZ 9.0', '  DIAM 6.5']
            + ['SURF 4', '  TYPE STANDARD', '  CURV 0.0', '  DISZ 5.0',
               '  DIAM 3.0']
            + ['SURF 5'] + img)


# ==========================================================================
# 1 -- the import
# ==========================================================================


class TestDgratingImport:
    """(a) A synthetic .zmx DGRATING round-trips period / order / PARM 1."""

    def test_period_and_order_round_trip(self):
        rx, _ = _load(_crossed_doe_zmx())
        d = rx['diffractives']
        assert len(d) == 2, f"expected 2 DGRATING surfaces, got {len(d)}"
        assert [x['surf_num'] for x in d] == [4, 6]
        # PARM 1 is LINES PER MICROMETRE, always -- not in the file's UNIT.
        for x in d:
            assert np.isclose(x['period'], 1e-6 / 0.00879, rtol=1e-12), (
                f"period {x['period']!r} is not 1e-6 / |lines_per_um|")
        assert d[0]['lines_per_um'] == 0.0087899999999999992
        assert d[1]['lines_per_um'] == -0.0087899999999999992
        # PARM 2 is the design diffraction order, kept as an int when integral
        assert d[0]['order'] == -4 and isinstance(d[0]['order'], int)
        assert d[1]['order'] == -2 and isinstance(d[1]['order'], int)
        assert d[0]['type'] == 'grating'
        # and the physical consequence: one order is lambda/period of tilt
        assert np.isclose(LAM / d[0]['period'], 1.31e-6 * 0.00879 * 1e6,
                          rtol=1e-12)

    def test_coordbrk_roll_and_sign_set_the_azimuth(self):
        """The crossed pair is crossed BECAUSE of the +90 deg z-roll between
        the two DGRATINGs; without folding it in, both gratings would fan
        along x and the whole 2-D lattice would collapse onto one axis."""
        rx, _ = _load(_crossed_doe_zmx())
        d = rx['diffractives']
        assert d[0]['angle_deg'] == 0.0
        # +90 from the COORDBRK, +180 from the negative lines/um (the sign
        # is the grating vector's direction; ``period`` stays positive)
        assert d[1]['angle_deg'] == 270.0
        assert d[0]['period'] > 0.0 and d[1]['period'] > 0.0
        # the deflections are therefore ORTHOGONAL, which is the point
        _, _, _, dL0, dM0, _, _, _ = _normalise_doe_entry(
            {'doe': d[0], 'order': 1}, 0, LAM, 'test')
        _, _, _, dL1, dM1, _, _, _ = _normalise_doe_entry(
            {'doe': d[1], 'order': 1}, 1, LAM, 'test')
        assert dM0 == 0.0 and dL0 > 0.0
        assert dL1 == 0.0 and dM1 < 0.0
        assert abs(dL0) == abs(dM1)

    def test_gaps_to_the_neighbouring_powered_elements(self):
        """THE bookkeeping the roadmap asks for: the 51.539 mm gap arrives
        with the DOE instead of being folded into a neighbour by hand.  Both
        air-air dummy planes (the STOP plane before, the reference plane
        after) collapse into the gaps, exactly as a consumer does by hand."""
        rx, _ = _load(_crossed_doe_zmx())
        d0, d1 = rx['diffractives']
        assert np.isclose(d0['gap_before'], 51.5393280925041e-3, rtol=1e-12)
        assert d0['gap_after'] == 0.0        # straight into the second DOE
        assert d1['gap_before'] == 0.0
        assert np.isclose(d1['gap_after'], 7.0e-3, rtol=1e-12)
        # and they add up to the hand-folded gap of the DOE-free control
        rx2, _ = _load(_plain_doublet_zmx())
        folded = d0['gap_before'] + d0['gap_after'] + d1['gap_before'] \
            + d1['gap_after']
        assert np.isclose(folded, rx2['thicknesses'][1], rtol=1e-12), (
            "the DOE's own gaps must sum to the gap a consumer folds by hand")

    def test_geometry_import_is_unchanged(self):
        """A DGRATING is still a FLAT optical surface: no aspheric
        coefficients from its PARM table (the pre-fix P2-19 failure), no
        freeform keys, and nothing diffractive on the lens-only 'surfaces'
        list that apply_real_lens / surfaces_from_prescription consume."""
        rx, _ = _load(_crossed_doe_zmx())
        by_num = {e['surf_num']: e for e in rx['elements']}
        for sn in (4, 6):
            e = by_num[sn]
            assert e['aspheric_coeffs'] is None
            assert e.get('freeform_type') is None
            assert not np.isfinite(e['radius'])      # CURV 0.0 -> flat
            assert e['glass_before'] == 'air' and e['glass_after'] == 'air'
        assert not any('diffractive' in s for s in rx['surfaces']), (
            "the lens-only prescription must stay free of diffractive keys "
            "-- nothing on that path diffracts")
        # the element's dict and the top-level list are the SAME object, so
        # they cannot drift apart
        assert by_num[4]['diffractive'] is rx['diffractives'][0]

    def test_warning_names_the_per_order_route(self):
        rx, msgs = _load(_crossed_doe_zmx())
        hits = [m for m in msgs if 'DGRATING' in m]
        assert len(hits) == 2, f"expected one warning per DGRATING: {msgs}"
        for m in hits:
            assert 'propagate_traced_carrier_chain' in m
            assert "diffractives" in m
            assert 'NO diffraction is applied' in m

    def test_dgrating_without_lines_per_um_attaches_nothing(self):
        lines = [ln for ln in _crossed_doe_zmx()
                 if 'PARM 1 0.0087899999999999992' not in ln]
        rx, msgs = _load(lines)
        assert len(rx['diffractives']) == 1          # only the second one
        assert rx['diffractives'][0]['surf_num'] == 6
        assert any('no usable PARM 1' in m for m in msgs)

    def test_no_dgrating_is_additive_only(self):
        """(c) A prescription WITHOUT a DGRATING is untouched: the only new
        key is an empty 'diffractives', and every pre-existing value still
        reads exactly what the loader contract says it should."""
        rx, msgs = _load(_plain_doublet_zmx())
        assert rx['diffractives'] == []
        assert not any('DGRATING' in m or 'diffractive' in m.lower()
                       for m in msgs)
        assert set(rx) == {
            'name', 'aperture_diameter', 'surfaces', 'thicknesses',
            'stop_index', 'elements', 'all_thicknesses', 'object_distance',
            'coord_breaks', 'diffractives'}, (
            "load_zemax_zmx's returned key set must only GAIN 'diffractives'")
        # pre-existing contract, recomputed independently from the file
        assert np.isclose(rx['surfaces'][0]['radius'], (1.0 / 0.01) * 1e-3)
        assert np.isclose(rx['surfaces'][0]['conic'], -0.5)
        assert np.isclose(rx['surfaces'][0]['aspheric_coeffs'][4],
                          1.5e-05 * 1e9, rtol=1e-12)
        assert np.isclose(rx['thicknesses'][1], 58.5393280925041e-3,
                          rtol=1e-12)
        assert len(rx['surfaces']) == 4 and len(rx['elements']) == 4

    def test_doe_spec_drops_straight_into_a_chain_entry(self):
        """The whole point of the import: no hand-built grating, no
        hand-typed gap.  ``{'doe': rx['diffractives'][k]}`` must validate and
        carry both gaps and the design order by itself."""
        rx, _ = _load(_crossed_doe_zmx())
        name, gb, ga, dL, dM, amp, org, order = _normalise_doe_entry(
            {'doe': rx['diffractives'][0]}, 0, LAM, 'test')
        assert np.isclose(gb, 51.5393280925041e-3, rtol=1e-12)
        assert ga == 0.0
        assert order == (-4.0, 0.0)          # the .zmx's own design order
        assert np.isclose(dL, -4 * LAM / rx['diffractives'][0]['period'],
                          rtol=1e-15)
        assert dM == 0.0 and amp == 1.0 and org == (0.0, 0.0)
        assert 'Diffractive' in name


class TestImportedGapsTileTheAxisOnce:
    """The drop-in contract's arithmetic: the chain transports
    ``gap_before + gap_after`` for EVERY DOE entry it is handed, so the
    loader must record each axial leg exactly once across the diffractives.

    An adversarial re-measure caught it recording the inter-DOE leg TWICE
    (once as DOE_k's ``gap_after``, once as DOE_(k+1)'s ``gap_before``,
    because a diffractive is an anchor for the neighbour scan), which made
    the documented drop-in transport 47 mm on a 37 mm .zmx.  Design 121's
    crossed pair is at DISZ 0.0 and could never expose it."""

    def test_the_inter_doe_leg_is_recorded_exactly_once(self):
        rx, _ = _load(_separated_doe_zmx())
        d0, d1 = rx['diffractives']
        assert [d['surf_num'] for d in rx['diffractives']] == [3, 4]
        # gap_before is always the TRUE distance from the previous element
        assert np.isclose(d0['gap_before'], SEP_T1 * 1e-3, rtol=1e-12)
        assert np.isclose(d1['gap_before'], SEP_T2 * 1e-3, rtol=1e-12)
        # gap_after is the TRAILING leg -- 0 when the next element is another
        # DGRATING, whose gap_before already carries it
        assert d0['gap_after'] == 0.0
        assert np.isclose(d1['gap_after'], SEP_T3 * 1e-3, rtol=1e-12)
        # THE invariant: what the chain transports is the .zmx's own distance
        total = sum(d['gap_before'] + d['gap_after']
                    for d in rx['diffractives'])
        assert np.isclose(total, (SEP_T1 + SEP_T2 + SEP_T3) * 1e-3,
                          rtol=1e-12), (
            f"the DOE entries transport {total * 1e3:.3f} mm across a "
            f"{SEP_T1 + SEP_T2 + SEP_T3:.3f} mm .zmx leg")
        # and it is the same total the loader's own thickness table carries
        # between the two glass surfaces that bracket the pair
        assert np.isclose(total, sum(rx['all_thicknesses'][1:4]), rtol=1e-12)

    def test_a_lone_dgrating_still_measures_both_neighbours(self):
        """The dedup must not fire when there is no second DGRATING: a single
        DOE keeps the true distance on BOTH sides."""
        lines = list(_separated_doe_zmx())
        cut = lines.index('SURF 4')           # SURF + COMM/TYPE/CURV/2x
        del lines[cut:lines.index('SURF 5')]  # PARM/DISZ/DIAM -> DOE-B gone
        rx, _ = _load(lines)
        assert len(rx['diffractives']) == 1
        d = rx['diffractives'][0]
        assert np.isclose(d['gap_before'], SEP_T1 * 1e-3, rtol=1e-12)
        assert np.isclose(d['gap_after'], SEP_T2 * 1e-3, rtol=1e-12)


class TestDgratingOutsideTheGlassSpan:
    """A DGRATING is an air-to-air flat, so the lens-window auto-detect has
    to count it as an active surface.  Before it did not: the window clipped
    to the glass span and any DOE outside it was discarded -- silently, and
    before the diffractive collector ran, so ``'diffractives'`` came back
    EMPTY with no warning.  Both layouts here are ordinary DOE designs."""

    @pytest.mark.parametrize('where, gb_mm, ga_mm',
                             [('first', 0.0, 12.0), ('last', 12.0, 0.0)])
    def test_it_is_imported_with_its_gaps(self, where, gb_mm, ga_mm):
        rx, msgs = _load(_doe_outside_glass_zmx(where))
        assert len(rx['diffractives']) == 1, (
            f"a DGRATING placed {where} in the file was dropped: "
            f"diffractives={rx['diffractives']}")
        d = rx['diffractives'][0]
        assert np.isclose(d['period'], 1e-6 / 0.00879, rtol=1e-12)
        assert d['order'] == -4
        assert np.isclose(d['gap_before'], gb_mm * 1e-3, rtol=1e-12)
        # 'last': there is no following anchor, so the remaining distance to
        # the image plane is the consumer's final_distance, exactly as for a
        # trailing lens surface
        assert np.isclose(d['gap_after'], ga_mm * 1e-3, rtol=1e-12)
        assert any('DGRATING' in m for m in msgs)
        # the element is in the imported window, carrying the same dict
        by_num = {e['surf_num']: e for e in rx['elements']}
        assert by_num[d['surf_num']]['diffractive'] is d

    def test_the_leading_gap_moves_out_of_object_distance(self):
        """A leading DOE also fixes where its 12 mm lives: before, the DOE
        was outside the window and the distance was swallowed by
        ``object_distance``; now it is an ordinary thickness that the DOE's
        own ``gap_after`` carries.  Either way no distance is invented."""
        rx, _ = _load(_doe_outside_glass_zmx('first'))
        d = rx['diffractives'][0]
        assert rx['object_distance'] == 0.0
        assert np.isclose(rx['object_distance'] + d['gap_before']
                          + d['gap_after'], 12.0e-3, rtol=1e-12)
        assert np.isclose(rx['all_thicknesses'][0], 12.0e-3, rtol=1e-12)

    def test_a_dgrating_without_grating_data_does_not_widen_the_window(self):
        """The window widens for DIFFRACTIVES, not for every surface that
        says DGRATING: one with no usable PARM 1 carries no grating at all
        and stays the plain flat plane it was, outside the glass span."""
        rx, msgs = _load(_doe_outside_glass_zmx('first', lines_per_um=None))
        assert rx['diffractives'] == []
        assert [e['surf_num'] for e in rx['elements']] == [2, 3]
        assert np.isclose(rx['object_distance'], 12.0e-3, rtol=1e-12)
        assert not any('no usable PARM 1' in m for m in msgs), (
            "the surface is outside the imported window, so it is not "
            "imported at all and must not warn about its PARM table")

    def test_an_explicit_surface_range_that_drops_one_warns(self):
        """An explicit ``surface_range`` is the caller's own window and is
        honoured -- but a DGRATING it excludes is exactly the "the design the
        chain sees has never contained the DOE" state, so say so."""
        fd, path = tempfile.mkstemp(suffix='.zmx', text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('\n'.join(_separated_doe_zmx()) + '\n')
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter('always')
                rx = load_zemax_zmx(path, surface_range=(1, 2))
        finally:
            os.unlink(path)
        msgs = [str(w.message) for w in wlist]
        assert rx['diffractives'] == []
        hit = [m for m in msgs if 'surface_range' in m and 'DGRATING' in m]
        assert hit, f"excluded DGRATINGs must be named: {msgs}"
        assert '[3, 4]' in hit[0]


# ==========================================================================
# 2 -- the grating equation and the entry contract (no propagation)
# ==========================================================================


class TestDoeEntryContract:

    def _spec(self, **kw):
        s = {'type': 'grating', 'period': 200e-6, 'order': 0,
             'angle_deg': 0.0, 'origin': (0.0, 0.0),
             'gap_before': 1e-3, 'gap_after': 2e-3}
        s.update(kw)
        return s

    def test_direction_cosine_shift_is_exact(self):
        """The grating equation shifts the TRANSVERSE WAVEVECTOR by the
        grating vector, so in direction cosines it is exact (not paraxial):
        ``L -> L + m lambda / P``."""
        for m in (-4, -0.5, 0, 1, 7):
            _, _, _, dL, dM, _, _, _ = _normalise_doe_entry(
                {'doe': self._spec(), 'order': m}, 0, LAM, 'test')
            assert dL == m * LAM / 200e-6
            assert dM == 0.0

    def test_crossed_grating_and_azimuth(self):
        _, _, _, dL, dM, _, _, _ = _normalise_doe_entry(
            {'doe': self._spec(period=(200e-6, 400e-6)), 'order': (2, 3)},
            0, LAM, 'test')
        assert dL == 2 * LAM / 200e-6
        assert dM == 3 * LAM / 400e-6
        # a 90 deg azimuth rotates the pair EXACTLY (no 1e-16 leak on the
        # quadrant multiples, which would drag a scalar chain onto the
        # tilted path for nothing)
        _, _, _, dL9, dM9, _, _, _ = _normalise_doe_entry(
            {'doe': self._spec(period=(200e-6, 400e-6), angle_deg=90.0),
             'order': (2, 3)}, 0, LAM, 'test')
        assert dL9 == -dM
        assert dM9 == dL

    def test_entry_overrides_spec(self):
        e = {'doe': self._spec(), 'order': 5, 'gap_before': 9e-3,
             'gap_after': 4e-3, 'amplitude': 0.5 - 0.25j, 'name': 'mine'}
        name, gb, ga, dL, dM, amp, _org, order = _normalise_doe_entry(
            e, 0, LAM, 'test')
        assert (name, gb, ga) == ('mine', 9e-3, 4e-3)
        assert order == (5.0, 0.0) and amp == 0.5 - 0.25j
        assert dL == 5 * LAM / 200e-6

    @pytest.mark.parametrize('entry,frag', [
        ({'doe': {'period': 1e-4}, 'thickness': 1e-3}, 'unknown key'),
        ({'doe': {'period': 1e-4, 'orders': [1, 2]}}, 'unknown key'),
        ({'doe': 1e-4}, 'must be a DOE spec dict'),
        ({'doe': {}}, "must supply 'period'"),
        ({'doe': {'period': -1e-4}}, 'positive pitch'),
        ({'doe': {'period': 1e-4}, 'order': (1, 2)}, 'no y period'),
        ({'doe': {'period': 1e-4, 'type': 'zone_plate'}}, "'grating'"),
        ({'doe': {'period': 1e-4}, 'order': np.inf}, 'must be finite'),
        ({'doe': {'period': 1e-4}, 'gap_before': np.nan}, 'finite distance'),
    ])
    def test_bad_entries_raise(self, entry, frag):
        with pytest.raises(ValueError, match=frag):
            _normalise_doe_entry(entry, 0, LAM, 'test')


# ==========================================================================
# 3 -- the chain: bookkeeping equivalence
# ==========================================================================

N_GRID = 256
DX0 = 9.0e-6
W0 = 250e-6
G0 = 10e-3          # input plane -> group 1
D1 = 20e-3          # group 1 exit -> DOE plane
TDOE = 5e-3         # the DOE's own trailing gap
D2 = 8e-3           # -> group 2
FD = 30e-3          # group 2 exit -> target
PERIOD = 400e-6


def _singlet(f, name, t=3e-3, glass='N-BK7', semi=6e-3):
    """Equiconvex singlet of roughly focal length ``f`` (the exact value is
    irrelevant -- every oracle below reads the prescription, not ``f``)."""
    R = 2 * f * 0.517
    return {'name': name,
            'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': glass,
                 'semi_diameter': semi},
                {'radius': -R, 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': glass, 'glass_after': 'air',
                 'semi_diameter': semi}],
            'thicknesses': [t]}


G1 = _singlet(150e-3, 'G1')
G2 = _singlet(150e-3, 'G2')
DOE = {'type': 'grating', 'period': PERIOD, 'order': 0, 'angle_deg': 0.0,
       'origin': (0.0, 0.0), 'gap_before': D1, 'gap_after': TDOE}


def _env0():
    x = (np.arange(N_GRID) - N_GRID // 2) * DX0
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / W0 ** 2
                  ).astype(np.complex128)


def _run(groups, **kw):
    k = dict(r_in=np.inf, ray_subsample=4, final_distance=FD,
             final_leg='paraxial')
    k.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return chain(_env0(), groups, LAM, DX0, **k)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))


@pytest.fixture(scope='module')
def runs():
    """Every chain run this module needs, computed once (~2.5 s each)."""
    pre = [{'prescription': G1, 'gap_before': G0}]
    post = [{'prescription': G2, 'gap_before': D2}]
    m = 3
    dL = m * LAM / PERIOD
    out = {'m': m, 'dL': dL}
    out['no_doe'] = _run(pre + [{'prescription': G2,
                                 'gap_before': D1 + TDOE + D2}])
    out['inert'] = _run(pre + [{'doe': DOE, 'gap_before': 0.0,
                                'gap_after': 0.0, 'order': 0},
                               {'prescription': G2,
                                'gap_before': D1 + TDOE + D2}])
    out['order0'] = _run(pre + [{'doe': DOE}] + post)
    out['split_a'] = _run(pre + [{'doe': DOE, 'order': m}] + post)
    out['split_b'] = _run(
        pre + [{'doe': DOE, 'gap_after': 0.0, 'order': m},
               {'prescription': G2, 'gap_before': TDOE + D2}])
    # the hand-split reference: chain A to the DOE plane, deflect by hand,
    # chain B onward -- what validation/repro_traced_carrier_121/
    # fan_multi_121.py does today, one order at a time.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tag, cref, tk in (('', 'sphere', {}),
                              ('_par', 'parabola',
                               {'traced_kwargs': {
                                   'amplitude_model': 'screen',
                                   'preserve_input_phase': True}})):
            a = chain(_env0(), pre, LAM, DX0, r_in=np.inf, ray_subsample=4,
                      final_distance=D1, final_leg='paraxial',
                      carrier_reference=cref, **tk)
            env_a = carrier_referenced_envelope(a.field, a.R, LAM, a.dx)
            b = chain(env_a,
                      [{'prescription': G2, 'gap_before': TDOE + D2}], LAM,
                      a.dx, r_in=la.TiltedCarrier(a.R, dL, 0.0, 0.0, 0.0),
                      ray_subsample=4, final_distance=FD,
                      final_leg='paraxial', carrier_reference=cref, **tk)
            out['manual' + tag] = b
            out['doe' + tag] = (
                out['split_a'] if not tag else
                _run(pre + [{'doe': DOE, 'order': m}] + post,
                     carrier_reference=cref, **tk))
    return out


class TestDoeChainBookkeeping:

    def test_zero_order_doe_is_inert(self, runs):
        """An order-0, unit-amplitude, zero-gap DOE inserted between two
        groups changes NOTHING -- the additive-only guarantee, and the proof
        that the new pending-gap bookkeeping is a no-op without a DOE."""
        a = np.asarray(runs['no_doe'].field)
        b = np.asarray(runs['inert'].field)
        scale = float(np.max(np.abs(a)))
        assert float(np.max(np.abs(a - b))) <= 1e-10 * scale
        assert runs['inert'].R == runs['no_doe'].R
        assert runs['inert'].dx == runs['no_doe'].dx

    def test_gap_after_folds_into_the_next_group(self, runs):
        """(b) THE bookkeeping equivalence, in its purest form: whether the
        DOE's trailing gap is declared on the DOE or on the next group's
        gap_before cannot matter.  Both segments are travelled at the same
        post-DOE angle, so the chain spends them as one leg -- bitwise."""
        a = np.asarray(runs['split_a'].field)
        b = np.asarray(runs['split_b'].field)
        scale = float(np.max(np.abs(a)))
        assert float(np.max(np.abs(a - b))) <= 1e-10 * scale
        assert runs['split_a'].dx == runs['split_b'].dx

    def test_order_zero_matches_the_hand_folded_gap(self, runs):
        """An UNDEFLECTED DOE is exactly a gap, so writing one into the design
        must reproduce the hand fold that a consumer does today -- BITWISE,
        because the envelope crosses the DOE plane inside one transport step
        (the order's action in the tracking frame is a constant, which
        commutes with the transport).  This is what guarantees that expressing
        a design's DOE cannot move a validated relay result: a SPLIT leg would
        agree only to ~1e-11 here, and not even that where the split plane
        lands near the carrier's own focus -- both measured in
        ``TestSplitLegPathDependence`` below."""
        a = np.asarray(runs['order0'].field)
        b = np.asarray(runs['no_doe'].field)
        assert float(np.max(np.abs(a - b))) <= 1e-10 * float(np.max(np.abs(a)))
        assert runs['order0'].dx == runs['no_doe'].dx

    def test_bitwise_survives_gaps_that_do_not_re_associate(self):
        """"Bitwise inert" has to be a property of the CHAIN, not luck about
        the numbers.  Float addition is not associative, so accumulating the
        deferred gaps pairwise -- ``(gb1+ga1) + (gb2+ga2)`` -- lands on a
        different float from the axial-order hand fold for perfectly ordinary
        gaps: 0.02/0.0/0.01/0.007 fold to 0.037 but pair up to
        0.037000000000000005.  One ulp is not small here -- a few ulp on a
        gap is enough to reach the traced pipeline's roundoff noise floor of
        ~1e-7 relative (measured: +1 ulp on this 37 mm gap gives 6.5e-11 on
        this relay and 1.4e-7 on a faster one; +10 ulp gives 8.1e-8 here) --
        so the pairwise sum turned "bitwise" into a measurable difference.
        The chain therefore accumulates ONE LEG AT A TIME, left to right,
        which is bit-identical to the axial-order fold for any gaps at
        all."""
        gb1, ga1, gb2, ga2 = 0.02, 0.0, 0.01, 0.007
        folded = gb1 + ga1 + gb2 + ga2
        assert (gb1 + ga1) + (gb2 + ga2) != folded, (
            "these gaps re-associate exactly, so this test proves nothing -- "
            "pick numbers whose pairwise sum differs in the last bit")
        spec = dict(DOE, order=0)
        a = _run([{'prescription': G1, 'gap_before': G0},
                  {'doe': spec, 'gap_before': gb1, 'gap_after': ga1},
                  {'doe': dict(spec), 'gap_before': gb2, 'gap_after': ga2},
                  {'prescription': G2, 'gap_before': 0.0}])
        b = _run([{'prescription': G1, 'gap_before': G0},
                  {'prescription': G2, 'gap_before': folded}])
        assert np.array_equal(np.asarray(a.field), np.asarray(b.field)), (
            f"not bitwise: rel {_rel(a.field, b.field):.3e}")
        # and the discriminator: one ulp on that gap -- 6.9e-18 m -- really
        # does change the answer, so re-associating the gap sum is not a
        # harmless refactor and "bitwise" is a claim with teeth
        c = _run([{'prescription': G1, 'gap_before': G0},
                  {'prescription': G2,
                   'gap_before': float(np.nextafter(folded, 1.0))}])
        assert not np.array_equal(np.asarray(b.field), np.asarray(c.field))
        assert _rel(b.field, c.field) > 1e-12, (
            "a 1-ulp gap change is invisible here, so bitwise is not the "
            "meaningful claim this test says it is")

    def test_matches_the_manual_hand_split(self, runs):
        """(b) The binding equivalence for a DEFLECTED order: the DOE entry
        reproduces the two-chain hand split (chain to the DOE plane, re-seed
        a TiltedCarrier by hand, chain onward) that design 121's fan runner
        performs today.

        The two differ only in what the HAND SPLIT does extra, and both terms
        are on its side of the ledger: it transports the leg in two pieces
        instead of one (measured 3.1e-8 under ``carrier_reference=
        'parabola'``, where that is the only difference), and under the
        shipped ``'sphere'`` default it additionally converts the envelope to
        the parabola reference and back at the split plane (5.3e-7)."""
        assert _rel(runs['doe_par'].field, runs['manual_par'].field) < 1e-6
        assert _rel(runs['doe'].field, runs['manual'].field) < 1e-4

    def test_chief_ray_matches_an_exact_meridional_raytrace(self, runs):
        """Independent oracle for what the bookkeeping actually claims: the
        deflected chief ray's landing position at the target.

        Traced here by hand -- exact Snell refraction at G2's two spherical
        surfaces, exact obliquity ``z L / sqrt(1 - L^2)`` on the free legs --
        with nothing from the library but the glass index.  This is what a
        manual fold gets WRONG (see the next test): the fold sends the chief
        ray over the whole folded distance at the post-DOE angle."""
        m, dL = runs['m'], runs['dL']
        n_g = float(la.get_glass_index('N-BK7', LAM))
        R1 = G2['surfaces'][0]['radius']
        R2 = G2['surfaces'][1]['radius']
        t = G2['thicknesses'][0]
        # chief ray: on axis through G1 (symmetry), deflected at the DOE
        x, u = 0.0, dL                       # (height, direction cosine)
        x += (TDOE + D2) * u / np.sqrt(1.0 - u * u)
        for (Rs, zv, n_in, n_out) in ((R1, 0.0, 1.0, n_g),
                                      (R2, t, n_g, 1.0)):
            # exact meridional refraction at a sphere of radius Rs whose
            # vertex is at z = zv (the ray is already at that vertex plane)
            #   intersection: solve |P + s d - C|^2 = Rs^2, C = (0, zv + Rs)
            d = np.array([u, np.sqrt(1.0 - u * u)])
            P = np.array([x, zv])
            C = np.array([0.0, zv + Rs])
            oc = P - C
            b = float(oc @ d)
            c = float(oc @ oc) - Rs * Rs
            disc = b * b - c
            assert disc > 0.0
            s = min((-b - np.sqrt(disc), -b + np.sqrt(disc)),
                    key=lambda ss: abs(float((P + ss * d)[1]) - zv))
            Q = P + s * d
            nv = np.sign(Rs) * (C - Q) / np.linalg.norm(C - Q)
            eta = n_in / n_out
            ci = float(d @ nv)
            d = eta * d + (np.sqrt(1.0 - eta * eta * (1.0 - ci * ci))
                           - eta * ci) * nv
            d = d / np.linalg.norm(d)
            # advance to the NEXT vertex plane (t for the first surface, the
            # exit vertex plane itself for the second)
            z_next = t if Rs is R1 else t
            if Rs is R1:
                s2 = (z_next - float(Q[1])) / float(d[1])
                Q = Q + s2 * d
            x, u = float(Q[0]), float(d[0])
            if Rs is R2:
                # back out to the exit VERTEX plane (z = t) exactly
                s2 = (t - float(Q[1])) / float(d[1])
                Q = Q + s2 * d
                x = float(Q[0])
        x += FD * u / np.sqrt(1.0 - u * u)
        got = float(runs['split_a'].stages[-1]['x_c'])
        assert runs['split_a'].stages[-1].get('target') is True
        assert abs(got - x) < 2e-3 * abs(x), (
            f"chain chief ray {got * 1e6:.3f} um vs exact meridional trace "
            f"{x * 1e6:.3f} um (order {m})")
        # and the library's own predictor (which places the multi
        # orchestrator's readout tiles) agrees with the chain to the digit
        pred = _chain_chief_ray_at_target(
            [{'prescription': G1, 'gap_before': G0},
             {'doe': DOE, 'order': m},
             {'prescription': G2, 'gap_before': D2}], LAM, np.inf, FD, 'test')
        assert abs(pred[0] - got) <= 1e-12 * abs(got)

    def test_the_naive_fold_is_measurably_wrong(self, runs):
        """The failure the feature removes.  Folding the DOE's gaps into a
        neighbour and re-seeding the tilt where the folded gap STARTS -- the
        arrangement a consumer reaches for, and the one this study got wrong
        once already -- loses the chief ray's advance over the POST-DOE
        segment, because the fold has the beam leave the previous group
        already deflected but still on axis."""
        dL = runs['dL']
        wrong = _run([{'prescription': G1, 'gap_before': G0},
                      {'prescription': G2, 'gap_before': D1 + TDOE + D2,
                       'r_in': la.TiltedCarrier(
                           runs['split_a'].stages[0]['R_out'], dL, 0.0,
                           0.0, 0.0)}])
        got = float(runs['split_a'].stages[-1]['x_c'])
        bad = float(wrong.stages[-1]['x_c'])
        assert abs(bad - got) > 50e-6, (
            "the naive fold must be visibly different, else this test is not "
            "discriminating anything")
        # ... and it is wrong by EXACTLY the lost post-DOE advance,
        # (gap_after + next gap_before) * L, imaged by G2 + the final leg
        A, B, C, D = la.system_abcd_prescription(G2, LAM)[0].ravel()
        lost = dL * (TDOE + D2) / np.sqrt(1.0 - dL * dL)
        pred_err = (A + FD * C) * lost
        assert abs(abs(bad - got) - abs(pred_err)) < 0.02 * abs(pred_err), (
            f"measured fold error {abs(bad - got) * 1e6:.3f} um vs predicted "
            f"{abs(pred_err) * 1e6:.3f} um")

    def test_grating_translation_by_one_period_is_a_symmetry(self):
        """Physical check on the DOE's phase reference -- the constant the
        chief-ray-tracking frame needs (``exp(i k (dL (x_c - ox) + ...))``)
        so that K orders recombine coherently at the image plane.

        A grating is periodic, so sliding its origin by one full period must
        leave an integer order's field unchanged, and by a half period must
        flip the sign of an odd order.  Placed here AFTER the last group,
        where everything downstream (a free leg, the reconstruct) is exactly
        linear in the field: through a traced element the same check holds
        only to ~5e-5, because ``apply_real_lens_traced``'s
        ``preserve_input_phase='remap'`` residual fit is itself weakly
        dependent on ABSOLUTE phase (measured on this relay: a global -1 on
        the input moves the output by 5.0e-5 relative, a global e^{0.7i} by
        1.7e-5 -- a wrapped-phase branch, unrelated to the DOE)."""
        m = 3
        pre = [{'prescription': G1, 'gap_before': G0},
               {'prescription': G2, 'gap_before': D1 + TDOE + D2}]
        base = dict(DOE, order=m, gap_before=0.0, gap_after=0.0)
        a = _run(pre + [{'doe': base}])
        b = _run(pre + [{'doe': dict(base, origin=(PERIOD, 0.0))}])
        c = _run(pre + [{'doe': dict(base, origin=(0.5 * PERIOD, 0.0))}])
        fa, fb, fc = (np.asarray(r.field) for r in (a, b, c))
        scale = float(np.max(np.abs(fa)))
        assert float(np.max(np.abs(fa - fb))) <= 1e-10 * scale
        assert float(np.max(np.abs(fa + fc))) <= 1e-10 * scale

    def test_trailing_doe_folds_into_final_distance(self):
        """A DOE after the LAST lens group still owns its gaps: with order 0
        they simply become part of the distance to the target."""
        pre = [{'prescription': G1, 'gap_before': G0},
               {'prescription': G2, 'gap_before': D1 + TDOE + D2}]
        a = _run(pre, final_distance=FD)
        b = _run(pre + [{'doe': dict(DOE, order=0), 'gap_before': 4e-3,
                         'gap_after': 6e-3}],
                 final_distance=FD - 10e-3)
        assert _rel(a.field, b.field) < 1e-5
        assert b.stages[-1]['doe'] is True

    def test_trailing_doe_with_the_exact_final_leg_raises(self):
        pre = [{'prescription': G1, 'gap_before': G0},
               {'prescription': G2, 'gap_before': D1 + TDOE + D2}]
        with pytest.raises(NotImplementedError, match='niche D4'):
            _run(pre + [{'doe': dict(DOE, order=0)}],
                 final_leg='exact',
                 focus_readout={'dx_out': 1e-6, 'N_out': 32})

    def test_doe_only_groups_raises(self):
        with pytest.raises(ValueError, match='only DOE entries'):
            _run([{'doe': DOE}])

    def test_stage_reports_the_order_it_ran(self, runs):
        st = [s for s in runs['split_a'].stages if s.get('doe')]
        assert len(st) == 1
        assert st[0]['order'] == (float(runs['m']), 0.0)
        assert st[0]['dL'] == runs['dL'] and st[0]['dM'] == 0.0
        assert st[0]['gap_before'] == D1 and st[0]['gap_after'] == TDOE


# ==========================================================================
# 3a -- the drop-in transports the file's distance, and no other
# ==========================================================================


class TestImportedDropInTransport:
    """End of the documented drop-in: load a .zmx, put every
    ``rx['diffractives'][k]`` in ``groups``, and the chain must transport the
    file's own axial distance -- no more, no less -- for the deflected orders
    as well as for order 0.

    Both halves of this were broken by the same defect: the loader recorded
    the inter-DOE leg twice, so on the 37 mm layout of
    :func:`_separated_doe_zmx` the drop-in transported 47 mm.  That is
    invisible on design 121 (its pair is at DISZ 0.0) and was invisible to
    every other test in this file for the same reason."""

    def _groups(self, diffractives, order=0, extra_gap_after=None):
        entries = []
        for k, d in enumerate(diffractives):
            e = {'doe': d, 'order': order}
            if extra_gap_after is not None and k == 0:
                # what the pre-fix loader handed a consumer: the inter-DOE
                # leg declared on BOTH neighbours
                e['gap_after'] = extra_gap_after
            entries.append(e)
        # the group after the last DOE gets gap_before=0 -- that DOE's
        # gap_after already carries the leg (the documented contract)
        return ([{'prescription': G1, 'gap_before': G0}] + entries
                + [{'prescription': G2, 'gap_before': 0.0}])

    def test_order_zero_drop_in_is_the_hand_folded_gap_bitwise(self):
        """The whole design expressed as one object must be BITWISE the
        DOE-free chain whose single gap is the axial-order sum of the same
        legs.  Bitwise, not 1e-10: the traced pipeline's roundoff noise floor
        is ~1e-7, so anything short of bitwise cannot distinguish "provably
        unchanged" from "unchanged as far as this pipeline can report"."""
        rx, _ = _load(_separated_doe_zmx())
        d0, d1 = rx['diffractives']
        folded = (d0['gap_before'] + d0['gap_after']
                  + d1['gap_before'] + d1['gap_after'])
        assert np.isclose(folded, (SEP_T1 + SEP_T2 + SEP_T3) * 1e-3,
                          rtol=1e-12)
        a = _run(self._groups(rx['diffractives'], order=0))
        b = _run([{'prescription': G1, 'gap_before': G0},
                  {'prescription': G2, 'gap_before': folded}])
        assert np.array_equal(np.asarray(a.field), np.asarray(b.field)), (
            f"order-0 drop-in is not bitwise: rel {_rel(a.field, b.field):.3e}")
        assert a.dx == b.dx and a.R == b.R

    def test_the_double_counted_leg_would_be_visibly_wrong(self):
        """The discriminator: hand the same chain the gaps the PRE-FIX loader
        produced (DOE-A's gap_after = the inter-DOE leg, which DOE-B's
        gap_before already carries) and the field must move, so the bitwise
        test above is pinning something real."""
        rx, _ = _load(_separated_doe_zmx())
        good = _run(self._groups(rx['diffractives'], order=0))
        bad = _run(self._groups(rx['diffractives'], order=0,
                                extra_gap_after=SEP_T2 * 1e-3))
        assert _rel(good.field, bad.field) > 1e-2, (
            "a 10 mm excess transport must be visible in the field")

    def test_deflected_drop_in_lands_where_the_file_says(self):
        """The same statement on the chief ray, at the .zmx's own design
        orders: the drop-in must land where a chain given the file's
        distances by hand lands.  A double-counted 10 mm leg moved it by
        433 um (31.6%) when this was measured on the pre-fix loader."""
        rx, _ = _load(_separated_doe_zmx())
        drop = [{'prescription': G1, 'gap_before': G0}]
        drop += [{'doe': d} for d in rx['diffractives']]
        drop += [{'prescription': G2, 'gap_before': 0.0}]
        # the same design with every distance typed from the .zmx text
        hand = [{'prescription': G1, 'gap_before': G0},
                {'doe': dict(rx['diffractives'][0]),
                 'gap_before': SEP_T1 * 1e-3, 'gap_after': 0.0},
                {'doe': dict(rx['diffractives'][1]),
                 'gap_before': SEP_T2 * 1e-3, 'gap_after': SEP_T3 * 1e-3},
                {'prescription': G2, 'gap_before': 0.0}]
        xa, ya, _, _ = _chain_chief_ray_at_target(drop, LAM, np.inf, FD, 'p')
        xb, yb, _, _ = _chain_chief_ray_at_target(hand, LAM, np.inf, FD, 'p')
        assert xa == xb and ya == yb
        # and it is genuinely deflected in BOTH axes (orders -4 and -2 on
        # gratings 90 deg apart would be the interesting case; here both
        # gratings run along x, so check the deflection is non-trivial)
        assert abs(xa) > 100e-6


# ==========================================================================
# 3b -- WHY the DOE plane does not interrupt the carrier leg
# ==========================================================================


class TestSplitLegPathDependence:
    """The measured basis for deferring the DOE's transport, pinned here
    because the shipped justification was once cited WRONG (a "5.5x co-moving
    pitch split on design 121's own DOE leg" that does not exist: that leg is
    collimated, R = +703.6 m, and the two routes agree to 2.1e-11).

    Two facts, both on a bare carrier leg with no lens in sight:

    * splitting an ORDINARY leg is nearly -- but not exactly -- inert.  The
      co-moving magnification telescopes exactly,
      ``(R+z1+z2)/R = [(R+z1)/R][(R+z1+z2)/(R+z1)]``, so the pitches agree to
      the last bit while the fields differ at the extra FFT pair's level.
      That 1e-11 is what deferring the transport turns into 0.
    * splitting AT a plane inside the near-focus bridge zone is NOT inert at
      all: the bridge re-grids, and the routes land on different grids.  This
      is the case a DOE entry has to be safe for even though design 121 is
      not it."""

    LEG_1 = 51.5393e-3          # design 121's DOE gap_before, to scale
    LEG_2 = 7.0e-3              # ... and its gap_after

    def _env(self, n=256, dx=20e-6, w=1.5e-3):
        x = (np.arange(n) - n // 2) * dx
        return (np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
                       ).astype(np.complex128), dx)

    @pytest.mark.parametrize('R', [np.inf, 703.5912, -0.2, -0.045, -0.030,
                                   -0.010, -0.003])
    def test_split_is_inert_when_the_split_plane_is_clear_of_the_focus(self,
                                                                       R):
        """Including R = -3 to -45 mm, where the carrier's focus is INSIDE
        the leg -- crossing the focus is not what breaks the split."""
        env, dx = self._env()
        z1, z2 = self.LEG_1, self.LEG_2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            one = propagate_carrier_referenced(env, R, z1 + z2, LAM, dx)
            b1 = propagate_carrier_referenced(env, R, z1, LAM, dx)
            two = propagate_carrier_referenced(b1.env, b1.R, z2, LAM, b1.dx)
        assert one.dx == pytest.approx(two.dx, rel=1e-12), (
            f"co-moving pitch must telescope: {one.dx} vs {two.dx}")
        a, b = np.asarray(one.env), np.asarray(two.env)
        scale = float(np.max(np.abs(a)))
        err = float(np.max(np.abs(a - b))) / scale
        assert err < 1e-9, f"R={R}: split leg differs by {err:.3e}"
        # ... and NOT bitwise -- which is exactly why the chain defers the
        # DOE's transport instead of splitting there.
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize('R,min_ratio', [(-0.051, 3.0),
                                             (-0.05155, 30.0),
                                             (-0.0516, 10.0)])
    def test_split_at_the_focus_re_grids(self, R, min_ratio):
        """The corner the deferral protects against: the split plane sits
        inside the near-focus bridge zone (focus at 51.0 / 51.55 / 51.6 mm
        against a split at 51.5393 mm), the bridge re-grids, and the two
        routes land on co-moving grids a factor 5.4 / 278 / 49 apart."""
        env, dx = self._env()
        z1, z2 = self.LEG_1, self.LEG_2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            one = propagate_carrier_referenced(env, R, z1 + z2, LAM, dx)
            b1 = propagate_carrier_referenced(env, R, z1, LAM, dx)
            two = propagate_carrier_referenced(b1.env, b1.R, z2, LAM, b1.dx)
        ratio = max(one.dx / two.dx, two.dx / one.dx)
        assert ratio > min_ratio, (
            f"R={R * 1e3:.4f} mm: expected a re-grid, got pitch ratio "
            f"{ratio:.4f} ({one.dx * 1e6:.4f} vs {two.dx * 1e6:.4f} um)")

    def test_an_order_zero_doe_is_immune_to_that_corner(self):
        """... and the DOE entry inherits none of it: with the transport
        deferred, an order-0 DOE placed anywhere in the leg is bitwise the
        hand-folded gap, independent of where the split would have fallen."""
        pre = [{'prescription': G1, 'gap_before': G0}]
        base = _run(pre + [{'prescription': G2,
                            'gap_before': D1 + TDOE + D2}])
        for gb in (0.0, D1, D1 + TDOE + D2):
            got = _run(pre + [{'doe': dict(DOE, order=0), 'gap_before': gb,
                               'gap_after': 0.0},
                              {'prescription': G2,
                               'gap_before': D1 + TDOE + D2 - gb}])
            assert _rel(base.field, got.field) <= 1e-10, (
                f"order-0 DOE at gap_before={gb} moved the result")


# ==========================================================================
# 4 -- one order per congruence through the multi orchestrator
# ==========================================================================


class TestDoeMultiOrders:

    def test_per_congruence_doe_order_runs_the_fan(self):
        """The end-to-end shape of the full configuration: ONE groups list
        containing the DOE, one congruence per order.  Each order must land
        where its own chief ray does, i.e. the +m and -m frames straddle the
        axis symmetrically."""
        m = 2
        groups = [{'prescription': G1, 'gap_before': G0},
                  {'doe': DOE},
                  {'prescription': G2, 'gap_before': D2}]
        env = _env0()
        congruences = [{'field': env, 'doe_order': +m, 'name': f'+{m}'},
                       {'field': env, 'doe_order': -m, 'name': f'-{m}'}]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = la.propagate_traced_carrier_chain_multi(
                congruences, groups, LAM, DX0,
                output_grid={'dx_out': 4e-6, 'N_out': 256},
                final_distance=FD, ray_subsample=4, final_leg='paraxial',
                readout_tile=64, on_readout_clip='ignore')
        cx = [c['chief_ray'][0] for c in res.congruences]
        assert np.isclose(cx[0], -cx[1], rtol=1e-9), (
            f"+/-{m} must straddle the axis, got {cx}")
        # ... and at the position the DOE-free bookkeeping predicts
        pred = _chain_chief_ray_at_target(
            [{'prescription': G1, 'gap_before': G0},
             {'doe': dict(DOE, order=m)},
             {'prescription': G2, 'gap_before': D2}], LAM, np.inf, FD, 't')
        assert abs(cx[0] - pred[0]) <= 4e-6          # one output pixel
        assert res.field.shape == (256, 256)
        assert np.isfinite(res.field).all()

    @pytest.mark.parametrize('where', ['trailing', 'between'])
    def test_doe_amplitude_reaches_the_exit_power_accounting(self, where):
        """A DOE's order ``amplitude`` must land in ``power_exit`` wherever
        the screen sits in ``groups``.

        The multi orchestrator takes its window-independent exit power from
        the last chain stage that reports one.  A TRAILING DOE used to report
        none, so ``power_exit`` was the last LENS group's -- measured BEFORE
        the screen scaled the field -- while ``power_out`` (the readout) was
        measured after.  ``capture = power_out / power_exit`` then read
        |amplitude|^-2 too small (0.2497 instead of 0.9988 at amplitude 0.5)
        and ``on_readout_clip`` fired on bookkeeping rather than on a clipped
        halo -- a FALSE production alarm, and a hard failure at
        ``on_readout_clip='error'``.  ``throughput`` was wrong by the same
        factor.  Both placements are checked so the fix cannot drift."""
        groups = {
            'trailing': lambda a: [
                {'prescription': G1, 'gap_before': G0},
                {'prescription': G2, 'gap_before': D1 + TDOE + D2},
                {'doe': dict(DOE, order=0), 'gap_before': 0.0,
                 'gap_after': 0.0, 'amplitude': a}],
            'between': lambda a: [
                {'prescription': G1, 'gap_before': G0},
                {'doe': dict(DOE, order=0), 'amplitude': a},
                {'prescription': G2, 'gap_before': D2}],
        }[where]
        env = _env0()
        out = {}
        for amp in (1.0, 0.5):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                res = la.propagate_traced_carrier_chain_multi(
                    [{'field': env, 'name': 'c0'}], groups(amp), LAM, DX0,
                    output_grid={'dx_out': 2e-6, 'N_out': 256},
                    final_distance=FD, ray_subsample=4,
                    final_leg='paraxial', readout_tile=256)
            out[amp] = res.congruences[0]
            out[(amp, 'clip')] = [w for w in rec
                                  if 'readout window' in str(w.message)]
        # power_exit and throughput scale with |a|^2 ...
        assert out[0.5]['power_exit'] == pytest.approx(
            0.25 * out[1.0]['power_exit'], rel=1e-9)
        assert out[0.5]['throughput'] == pytest.approx(
            0.25 * out[1.0]['throughput'], rel=1e-9)
        # ... so capture -- a RATIO of two post-screen powers -- does not.
        assert out[0.5]['capture'] == pytest.approx(out[1.0]['capture'],
                                                    rel=1e-9)
        assert out[1.0]['capture'] > 0.9
        # and the clip guard sees the same thing at both amplitudes
        assert bool(out[(0.5, 'clip')]) == bool(out[(1.0, 'clip')])

    def test_doe_stage_reports_the_power_across_the_screen(self):
        """The single-chain half of the same contract: the DOE stage's
        ``power`` is the last lens exit power times |amplitude|^2 (the screen
        multiplies the envelope by a complex constant, so this is exact), and
        it is the value the multi's exit-power accounting picks up."""
        pre = [{'prescription': G1, 'gap_before': G0},
               {'prescription': G2, 'gap_before': D1 + TDOE + D2}]
        amp = 0.25 + 0.5j
        res = _run(pre + [{'doe': dict(DOE, order=0), 'gap_before': 0.0,
                           'gap_after': 0.0, 'amplitude': amp}])
        lens = [s for s in res.stages if not s.get('doe')][-1]
        doe = [s for s in res.stages if s.get('doe')][-1]
        assert doe['power'] == pytest.approx(
            lens['power'] * abs(amp) ** 2, rel=1e-12)
        assert _multi_chain_exit_power(res.stages) == doe['power']
        # the screen carries no grid state of its own (deferred transport)
        assert 'dx' not in doe and 'w' not in doe

    def test_doe_order_without_a_doe_raises(self):
        with pytest.raises(ValueError, match='no DOE entry'):
            la.propagate_traced_carrier_chain_multi(
                [{'field': _env0(), 'doe_order': 1}],
                [{'prescription': G1, 'gap_before': G0}], LAM, DX0,
                output_grid={'dx_out': 4e-6, 'N_out': 32},
                final_distance=FD, final_leg='paraxial')

    def test_unknown_congruence_key_still_raises(self):
        with pytest.raises(ValueError, match='unknown key'):
            la.propagate_traced_carrier_chain_multi(
                [{'field': _env0(), 'doe_orders': 1}],
                [{'prescription': G1, 'gap_before': G0}], LAM, DX0,
                output_grid={'dx_out': 4e-6, 'N_out': 32},
                final_distance=FD, final_leg='paraxial')
