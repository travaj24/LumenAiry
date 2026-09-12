"""WP-A20 regressions: the pre-U1 lens-library migration and the public
``rs_alias_free_distance`` (audit 2026-09-11 loose ends).

Two independent items, one file.

(1) ``user_library.load_lens`` back-fill  (WP-A9 report section 6.1,
    VERIFY-A9 section 5.3)
--------------------------------------------------------------------
A folded design saved to the lens library by the designer BEFORE the
U1/U2 fix has a chronological ``surfaces`` list -- its mirror IS in it --
but carries none of ``is_mirror`` / ``semi_diameter`` / ``is_stop`` on
those entries.  ``surfaces_from_prescription`` then reads the fold as
three refracting surfaces in air (an air->air no-op where a mirror
should be) and, because the mirror is missing from the refracting-only
``elements`` filter while the index counts it, hands the mirror the
FOLLOWING lens's aperture.

ORACLE.  The layout the entry was saved from: ``system_abcd`` of the
designer model's own ``build_trace_surfaces()`` list -- the list the 2-D
layout and the spot diagram are drawn from, sharing no code with the
prescription round trip past the ``Surface`` dataclass.  Both sides are
float64 evaluations of the same paraxial recursion on the same inputs,
so the bar is EXACT equality (``==``), not a tolerance: after the
back-fill the two agree to the last bit or they do not agree at all.

MEASURED, on the fixture built below (concave fold mirror R = -200 mm,
semi-diameter 25 mm, then an N-BK7 singlet R1 = 80 mm at a Zemax-signed
-30 mm), at the d line:

    layout / modern save        EFL 0.1825479198383674 m
                                BFL 0.1304206774044291 m
                                semi-diameters [0.025, 0.006, 0.006]
    pre-fix save, NO back-fill  EFL 0.15479922951064448 m   (-15.2 %)
                                BFL 0.1521620959930726 m    (+16.7 %)
                                semi-diameters [0.006, 0.006, 0.0127]
                                is_mirror [False, False, False]
    pre-fix save, back-filled   EXACTLY the layout numbers

The no-back-fill row is this file's FAIL-BEFORE: it is re-measured in
``test_a20_prefix_folded_entry_without_the_backfill_is_wrong`` by calling
``surfaces_from_prescription`` on the stripped dict directly, so the
pre-fix number is asserted here rather than quoted from a report.

The fixtures are NOT hand-written prescriptions: they are produced by the
CURRENT ``SystemModel.to_prescription()`` and then stripped of exactly
the three keys (and of ``stop_index``, and of ``is_stop`` on the
``elements`` entries, which the pre-fix exporter also did not write), so
the "pre-fix file" differs from the modern one in nothing else.

One residual the back-fill deliberately does NOT repair -- a coordinate
break's transfer thickness, which the same old exporter dropped in the
``cb_post`` case -- is reported instead, and
``test_a20_a_lost_coord_break_gap_is_reported_not_repaired`` pins both
the report and the non-repair with the pre-fix numbers.

(2) ``rs_alias_free_distance``  (WP-A15b report section 5.3)
------------------------------------------------------------
``2*N*dx**2/wavelength`` is the distance ``rayleigh_sommerfeld_propagate(
kernel='auto')`` branches on and the distance below which
``kernel='spatial'`` refuses, so it is the number a caller needs in order
to choose ``z`` / ``N`` / ``dx`` -- public API, not an internal.  Pinned
here: the closed form (exact, an oracle with no error floor: the same
three multiplications), the alias identity of the retained private name
(``is``, not equality), and that the function really is the branch point
(``kernel='auto'`` is bit-identical to ``'transfer'`` just below it and
to ``'spatial'`` at it).
"""
from __future__ import annotations

import copy
import importlib
import os
import pathlib
import sys
import tempfile
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                          # noqa: E402
import pytest                                               # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
_STUB = (_REPO / 'docs' / 'audits'
         / 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11' / 'repro' / 'UI' / 'stub')

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lumenairy import user_library as ul                    # noqa: E402
from lumenairy.io import load_zemax_zmx                     # noqa: E402
from lumenairy.propagators import rs as _rs                 # noqa: E402
from lumenairy.raytrace import (                            # noqa: E402
    surfaces_from_prescription,
    system_abcd,
)

#: d line.  Any wavelength works -- both sides of every comparison use the
#: same one -- but a catalogue glass wants a real one.
_WV = 587.6e-9

#: The keys the pre-U1 designer did not write onto ``surfaces``.
_KEYS = ('is_mirror', 'semi_diameter', 'is_stop')


# ===========================================================================
# Fixture capture: the designer's OWN exporter, under the auditor's Qt stub
# ===========================================================================
#
# PySide6 is not installed on this workstation and that is not a reason to
# skip (TESTING_STANDARDS S3): ``lumenairy/ui/model.py`` needs exactly
# ``PySide6.QtCore.QObject`` / ``Signal``, which the auditor's 60-line stub
# under repro/UI/stub provides.  A real PySide6 is preferred when present.
#
# The stub is installed, the two prescriptions and the layout ABCDs are
# captured as plain data, and then BOTH the stub and the ``lumenairy.ui``
# modules it built are lifted back out of ``sys.modules`` -- pytest imports
# every test module during collection, and sibling UI test files guard
# themselves with ``try: import PySide6 / except ImportError: skip``, so a
# stub left in place would bring them out of their skip and fail them
# against a Qt that cannot paint.  Nothing below this block needs Qt, so
# unlike ``test_audit2609_a9_ui.py`` there is no un-parking fixture.


def _capture_designer_fixtures():
    """Return ``(flavour, fixtures)`` captured from the real exporter."""
    flavour = 'real'
    try:
        import PySide6.QtCore  # noqa: F401
    except Exception:
        assert _STUB.is_dir(), (
            f'Qt stub missing at {_STUB}; it is the only harness for '
            f'lumenairy.ui on an interpreter without PySide6.')
        if str(_STUB) not in sys.path:
            sys.path.insert(0, str(_STUB))
        flavour = 'stub'

    before = set(sys.modules)
    try:
        model = importlib.import_module('lumenairy.ui.model')
        SystemModel = model.SystemModel
        Element = model.Element
        SurfaceRow = model.SurfaceRow

        def _folded():
            """A 45-degree-class fold: concave mirror R = -200 mm with a
            25 mm semi-diameter, then an N-BK7 singlet at a Zemax-signed
            -30 mm (the audit's own ``repro/UI/t2_mirror.py`` system)."""
            m = SystemModel()
            m.insert_element(1, Element(
                0, 'M1', 'Mirror', distance_mm=50.0,
                surfaces=[SurfaceRow(radius=-200.0, thickness=0.0, glass='',
                                     semi_diameter=25.0, surf_type='Mirror')]))
            m.insert_element(2, Element(
                0, 'Lens 2', 'Singlet', distance_mm=-30.0,
                surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                          SurfaceRow(np.inf, 0.0, '', 6.0)]))
            return m

        def _folded_with_stop():
            """The same fold with the aperture stop ON the mirror -- the
            only way ``is_stop`` reaches a mirror entry."""
            m = SystemModel()
            m.insert_element(1, Element(
                0, 'M1', 'Mirror', distance_mm=50.0,
                surfaces=[SurfaceRow(radius=-200.0, thickness=0.0, glass='',
                                     semi_diameter=25.0, surf_type='Mirror',
                                     is_stop=True)]))
            m.insert_element(2, Element(
                0, 'Lens 2', 'Singlet', distance_mm=-30.0,
                surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                          SurfaceRow(np.inf, 0.0, '', 6.0)]))
            return m

        def _folded_tilted():
            """A mirror with a TILTED element behind it -- the ``cb_post``
            case, the only topology in which the pre-fix exporter also
            dropped a coordinate break's transfer thickness."""
            m = SystemModel()
            m.insert_element(1, Element(
                0, 'M1', 'Mirror', distance_mm=50.0,
                surfaces=[SurfaceRow(radius=-200.0, thickness=0.0, glass='',
                                     semi_diameter=25.0, surf_type='Mirror')]))
            lens = Element(0, 'Lens 2', 'Singlet', distance_mm=40.0,
                           surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                                     SurfaceRow(np.inf, 0.0, '', 6.0)])
            lens.tilt_x = 45.0
            m.insert_element(2, lens)
            return m

        def _unfolded():
            """The control: an ordinary N-BK7 biconvex singlet, no mirror
            anywhere, so the migration must not touch it."""
            m = SystemModel()
            m.insert_element(1, Element(
                0, 'L1', 'Singlet', distance_mm=10.0,
                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                          SurfaceRow(-50.0, 0.0, '', 12.7)]))
            return m

        out = {}
        for name, build in (('folded', _folded),
                            ('folded_stop', _folded_with_stop),
                            ('folded_tilted', _folded_tilted),
                            ('unfolded', _unfolded)):
            mdl = build()
            _, efl, bfl, _ = system_abcd(mdl.build_trace_surfaces(), _WV)
            out[name] = {'rx': mdl.to_prescription(),
                         'layout_efl': float(efl),
                         'layout_bfl': float(bfl)}
        return flavour, out
    finally:
        if flavour == 'stub':
            for key in sorted(set(sys.modules) - before, reverse=True):
                if (key == 'PySide6' or key.startswith('PySide6.')
                        or key == 'lumenairy.ui'
                        or key.startswith('lumenairy.ui.')):
                    sys.modules.pop(key, None)
            if str(_STUB) in sys.path:
                sys.path.remove(str(_STUB))


QT_FLAVOUR, _FIX = _capture_designer_fixtures()


def _strip_to_prefix(rx):
    """Turn a modern designer export into the file a PRE-U1 designer wrote.

    The pre-fix ``to_prescription`` emitted the same ``surfaces`` dicts
    minus :data:`_KEYS`, the same ``elements`` dicts minus ``is_stop``,
    and no ``stop_index`` at all (verified against
    ``git show 58ad836c^:lumenairy/ui/model.py``).  Everything else --
    radii, conics, glasses, thicknesses, ``elements`` semi-diameters --
    is byte-for-byte what it writes today, which is what makes the
    comparison below a measurement of the back-fill and of nothing else.
    """
    out = copy.deepcopy(rx)
    for s in out['surfaces']:
        for k in _KEYS:
            s.pop(k, None)
    for e in out.get('elements', []):
        e.pop('is_stop', None)
    out.pop('stop_index', None)
    return out


@pytest.fixture()
def lib(tmp_path):
    """A private, empty user library for one test."""
    prev = ul._get_library_path_override()
    ul.set_library_path(str(tmp_path / 'lib'))
    try:
        yield
    finally:
        ul._set_library_path_override(prev)


def _efl_bfl_sd(rx):
    surfs = surfaces_from_prescription(rx)
    _, efl, bfl, _ = system_abcd(surfs, _WV)
    return (float(efl), float(bfl),
            [float(s.semi_diameter) for s in surfs],
            [bool(s.is_mirror) for s in surfs])


def _save_and_load(name, rx):
    ul.save_lens(name, rx)
    return ul.load_lens(name)


# ===========================================================================
# (1) the pre-U1 folded-entry back-fill
# ===========================================================================

class TestA20FoldedLibraryBackfill:

    def test_a20_prefix_folded_entry_without_the_backfill_is_wrong(self):
        """FAIL-BEFORE, re-measured rather than quoted.

        Resolve the stripped prescription through
        ``surfaces_from_prescription`` DIRECTLY -- the path
        ``load_lens`` took before this work package -- and show it
        disagrees with the layout the entry was saved from.  Everything
        the back-fill claims to repair is asserted here in its broken
        form, so a back-fill that silently stopped working would make the
        NEXT test fail while this one still passes: the two together pin
        the direction of the change, not just its presence.
        """
        rx = _FIX['folded']['rx']
        efl0, bfl0, sd0, mir0 = _efl_bfl_sd(_strip_to_prefix(rx))

        # No mirror anywhere: the fold analyses as an air->air surface.
        assert mir0 == [False, False, False], mir0
        # The refracting-only 'elements' filter shifts by one per mirror,
        # so the mirror gets the FOLLOWING lens's 6 mm and the last lens
        # surface falls back to aperture_diameter / 2.
        assert sd0 == [0.006, 0.006, 0.0127], sd0
        # 15.2 % / 16.7 % away from the layout, i.e. 13 decades outside
        # the 1e-15 relative floor two float64 evaluations of the same
        # recursion would sit inside.
        lay_efl = _FIX['folded']['layout_efl']
        lay_bfl = _FIX['folded']['layout_bfl']
        assert abs(efl0 / lay_efl - 1.0) > 0.10, (efl0, lay_efl)
        assert abs(bfl0 / lay_bfl - 1.0) > 0.10, (bfl0, lay_bfl)

    def test_a20_prefix_folded_entry_loads_as_the_layout_it_was_saved_from(
            self, lib):
        """The deliverable: a pre-fix folded entry, loaded, equals the
        modern save of the same design AND the designer's own layout --
        bit for bit, not to a tolerance."""
        rx = _FIX['folded']['rx']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            full = _save_and_load('a20_folded_modern', copy.deepcopy(rx))
            pre = _save_and_load('a20_folded_prefix', _strip_to_prefix(rx))

        e_full, b_full, sd_full, mir_full = _efl_bfl_sd(full)
        e_pre, b_pre, sd_pre, mir_pre = _efl_bfl_sd(pre)

        assert (e_pre, b_pre) == (e_full, b_full), (
            (e_pre, b_pre), (e_full, b_full))
        assert sd_pre == sd_full == [0.025, 0.006, 0.006], (sd_pre, sd_full)
        assert mir_pre == mir_full == [True, False, False], (mir_pre, mir_full)
        # ... and both equal the layout the designer drew.
        assert (e_full, b_full) == (_FIX['folded']['layout_efl'],
                                    _FIX['folded']['layout_bfl'])

    def test_a20_backfill_announces_itself(self, lib):
        """A load that changes a stored design's EFL by 15 % must say so."""
        pre = _strip_to_prefix(_FIX['folded']['rx'])
        ul.save_lens('a20_folded_warn', pre)
        with pytest.warns(UserWarning, match='back-filled'):
            ul.load_lens('a20_folded_warn')

    def test_a20_prefix_unfolded_entry_is_untouched(self, lib):
        """The control: an UNFOLDED pre-fix entry needs no repair, so it
        gets none -- the loaded dict is equal to the stored one key for
        key (no ``is_mirror`` appears anywhere), no warning fires, and its
        EFL/BFL are bit-identical to a modern save of the same design."""
        rx = _FIX['unfolded']['rx']
        stripped = _strip_to_prefix(rx)
        ul.save_lens('a20_unfolded_prefix', copy.deepcopy(stripped))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pre = ul.load_lens('a20_unfolded_prefix')
        assert not [w for w in caught
                    if 'load_lens' in str(w.message)], [str(w.message)
                                                        for w in caught]
        assert pre == stripped, 'an unfolded entry must load unchanged'
        for s in pre['surfaces']:
            for k in _KEYS:
                assert k not in s, (k, s)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            full = _save_and_load('a20_unfolded_modern', copy.deepcopy(rx))
        e_pre, b_pre, sd_pre, mir_pre = _efl_bfl_sd(pre)
        e_full, b_full, sd_full, _ = _efl_bfl_sd(full)
        assert (e_pre, b_pre) == (e_full, b_full) == (
            _FIX['unfolded']['layout_efl'], _FIX['unfolded']['layout_bfl'])
        assert sd_pre == sd_full == [0.0127, 0.0127], (sd_pre, sd_full)
        assert mir_pre == [False, False]

    def test_a20_backfill_recovers_the_stop_both_ways(self, lib):
        """``is_stop`` on the mirror reaches BOTH spellings.

        ``surfaces_from_prescription`` prefers the per-surface flag;
        ``apply_real_lens`` reads only ``prescription['stop_index']``.  An
        entry whose ``elements`` carry ``is_stop`` (the vintage between
        the two designer versions) must come back with both, at the same
        index the modern exporter writes.
        """
        rx = _FIX['folded_stop']['rx']
        assert rx['stop_index'] == 0, rx['stop_index']
        stripped = copy.deepcopy(rx)
        for s in stripped['surfaces']:
            for k in _KEYS:
                s.pop(k, None)
        stripped.pop('stop_index', None)          # elements keep is_stop
        ul.save_lens('a20_folded_stop', stripped)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            loaded = ul.load_lens('a20_folded_stop')
        assert [bool(s['is_stop']) for s in loaded['surfaces']] == [
            True, False, False]
        assert loaded['stop_index'] == 0
        assert [bool(s.is_stop) for s in surfaces_from_prescription(loaded)] \
            == [True, False, False]

    def test_a20_a_file_that_already_carries_is_mirror_is_never_rewritten(
            self, lib):
        """A modern save round-trips byte-for-byte and silently: the
        migration is for files that PREDATE the keys, and a producer that
        writes them is never second-guessed."""
        rx = copy.deepcopy(_FIX['folded']['rx'])
        ul.save_lens('a20_folded_modern2', copy.deepcopy(rx))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loaded = ul.load_lens('a20_folded_modern2')
        assert not [w for w in caught if 'load_lens' in str(w.message)]
        assert loaded == rx

    def test_a20_lens_only_zmx_shape_is_left_alone_and_silent(self, lib):
        """The shape the back-fill must NOT touch, taken from a real file.

        ``repro/IO-OPTIMIZE/cb_mirror.zmx`` is the lens-only layout every
        .zmx / CodeV loader produces: two refracting ``surfaces``, three
        ``elements`` of which one is the mirror.  The lengths differ, so
        surface *i* is the *i*-th ``element_type == 'surface'`` entry and
        the existing resolver is already right -- the trigger must not
        fire, and it must not warn either, because there is nothing wrong
        with this file.
        """
        zmx = _REPO / ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11'
                       '/repro/IO-OPTIMIZE/cb_mirror.zmx')
        assert zmx.is_file(), zmx
        rx = load_zemax_zmx(str(zmx))
        assert len(rx['elements']) != len(rx['surfaces'])
        assert any(e.get('element_type') == 'mirror' for e in rx['elements'])
        assert all('is_mirror' not in s for s in rx['surfaces'])

        before = _efl_bfl_sd(rx)
        ul.save_lens('a20_zmx_mirror', rx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loaded = ul.load_lens('a20_zmx_mirror')
        assert not [w for w in caught if 'load_lens' in str(w.message)], [
            str(w.message) for w in caught]
        for s in loaded['surfaces']:
            assert 'is_mirror' not in s, s
        assert _efl_bfl_sd(loaded) == before

    @pytest.mark.parametrize('corrupt, why', [
        ('radius', "elements[1]['radius']"),
        ('element_type', "element_type"),
        ('glass', "glass_before"),
    ])
    def test_a20_backfill_refuses_to_guess_an_inconsistent_pairing(
            self, lib, corrupt, why):
        """When the trigger fires but the two lists are NOT positionally
        consistent, nothing is back-filled and the refusal is AUDIBLE.

        Three ways the pairing can be wrong, each applied to an otherwise
        valid pre-fix folded entry: a radius that does not match its
        surface (a lens-only ``elements`` list that happens to have the
        same length -- the case VERIFY-A9 section 2.4 guards against), an
        ``elements`` entry that is not an optical surface at all, and a
        glass that disagrees.  In every case the loaded prescription must
        still lack the keys, so the caller gets the old (wrong but
        unchanged) numbers plus a warning naming the index, rather than a
        silently guessed aperture mapping.
        """
        pre = _strip_to_prefix(_FIX['folded']['rx'])
        if corrupt == 'radius':
            pre['elements'][1]['radius'] = pre['elements'][1]['radius'] * 2 + 1
        elif corrupt == 'element_type':
            # entry 0 stays the mirror (that is the trigger); entry 1 is
            # what an 'elements' list carrying non-optical rows looks like
            pre['elements'][1]['element_type'] = 'coord_break'
        else:
            pre['elements'][1]['glass_before'] = 'N-SF11'
        ul.save_lens(f'a20_bad_{corrupt}', pre)
        with pytest.warns(UserWarning, match='NOT positionally consistent'):
            loaded = ul.load_lens(f'a20_bad_{corrupt}')
        for s in loaded['surfaces']:
            for k in _KEYS:
                assert k not in s, (k, s)
        assert 'stop_index' not in loaded
        # Nothing at all was written: the loaded dict is key-for-key the
        # stored one, so there is no half-applied mixture to reason about.
        assert loaded == pre

    def test_a20_a_lost_coord_break_gap_is_reported_not_repaired(self, lib):
        """The residual the back-fill cannot close, made audible.

        The same pre-U1 exporter that omitted the three keys ALSO dropped
        a coordinate break's transfer thickness in the ``cb_post`` case (a
        tilted element behind a mirror).  Measured by exporting the
        fixture below through ``58ad836c^:lumenairy/ui/model.py`` loaded
        in-process alongside the current one:

            modern  thicknesses [0.04, 0.004]   all_thicknesses [0.04, 0.004, 0.1]
            pre-fix thicknesses [0.0,  0.004]   all_thicknesses [0.04, 0.004, 0.1]

        -- i.e. 40 mm of air vanished from ``thicknesses`` while
        ``all_thicknesses`` kept it.  The stored dict below reproduces the
        pre-fix export exactly (keys stripped AND ``thicknesses[0]``
        zeroed).  Loading it must warn a SECOND time, quoting both
        numbers, because the back-fill repairs the mirror and leaves the
        gap: layout EFL/BFL 1.0459951945424169 / 1.4670304058769554 m, no
        back-fill 0.15479922951064448 / 0.1521620959930726, back-filled
        but gap still missing 0.2824843175588363 / 0.2851214510764082.

        The untilted fold is the counter-pin: its pre-fix and modern
        ``thicknesses`` are identical (measured on nine topologies), so it
        must get the first warning and NOT this one.
        """
        pre = _strip_to_prefix(_FIX['folded_tilted']['rx'])
        assert len(pre['coord_breaks']) == 1, pre['coord_breaks']
        assert pre['thicknesses'] == [0.04, 0.004], pre['thicknesses']
        assert pre['all_thicknesses'] == [0.04, 0.004, 0.1]
        pre['thicknesses'][0] = 0.0               # what the old code wrote
        ul.save_lens('a20_folded_tilted_prefix', pre)
        with pytest.warns(UserWarning) as rec:
            loaded = ul.load_lens('a20_folded_tilted_prefix')
        msgs = [str(w.message) for w in rec]
        assert any('back-filled' in m for m in msgs), msgs
        gap = [m for m in msgs if 'coordinate break' in m]
        assert len(gap) == 1, msgs
        assert 'thicknesses[0]=0.0' in gap[0], gap[0]
        assert 'all_thicknesses[0]=0.04' in gap[0], gap[0]
        # reported, NOT repaired
        assert loaded['thicknesses'][0] == 0.0
        assert [bool(s['is_mirror']) for s in loaded['surfaces']] == [
            True, False, False]

        # counter-pin: the untilted fold has no lost gap and must not
        # raise the second warning.
        ul.save_lens('a20_folded_plain_prefix',
                     _strip_to_prefix(_FIX['folded']['rx']))
        with pytest.warns(UserWarning) as rec2:
            ul.load_lens('a20_folded_plain_prefix')
        assert not [w for w in rec2
                    if 'coordinate break' in str(w.message)], [
            str(w.message) for w in rec2]

    def test_a20_backfill_helper_tolerates_junk_without_raising(self):
        """``load_lens`` must never raise because of a malformed stored
        entry -- the migration either applies or stands aside."""
        for junk in (None, 42, {}, {'surfaces': []},
                     {'surfaces': [{'radius': 1.0}], 'elements': None},
                     {'surfaces': [{'radius': 1.0}],
                      'elements': [{'element_type': 'mirror'}]},
                     {'surfaces': ['not a dict'],
                      'elements': [{'element_type': 'mirror'}]}):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                assert ul._backfill_folded_surface_keys(
                    copy.deepcopy(junk), 'junk') is not Ellipsis

    def test_a20_same_radius_helper_matches_inf_and_never_nan(self):
        """The positional fingerprint's own edge cases.  Both numbers come
        from one ``radius_mm * 1e-3`` in the producer and JSON round-trips
        a float exactly, so equality is the right test -- but a flat
        surface is ``inf`` on both sides (must match) and a NaN radius
        must never be allowed to "match" itself into a mapping."""
        assert ul._same_radius(0.1, 0.1)
        assert ul._same_radius(float('inf'), float('inf'))
        assert ul._same_radius(None, None)
        assert not ul._same_radius(float('inf'), float('-inf'))
        assert not ul._same_radius(float('nan'), float('nan'))
        assert not ul._same_radius(0.1, 0.1 + 1e-18 + 1e-17)
        assert not ul._same_radius(0.1, None)


# ===========================================================================
# (2) rs_alias_free_distance is public
# ===========================================================================

class TestA20RsAliasFreeDistanceIsPublic:

    def test_a20_public_name_exists_and_is_exported(self):
        assert hasattr(_rs, 'rs_alias_free_distance')
        assert 'rs_alias_free_distance' in _rs.__all__
        assert '_rs_alias_free_distance' not in _rs.__all__
        fn = _rs.rs_alias_free_distance
        assert fn.__name__ == 'rs_alias_free_distance'
        assert fn.__qualname__ == 'rs_alias_free_distance'
        assert (fn.__doc__ or '').strip(), 'new public API needs a docstring'

    def test_a20_private_alias_is_the_same_object(self):
        """The three audit regression files import the private spelling;
        it stays bound to the SAME function object, so monkeypatching
        either name is monkeypatching one function and there is no
        wrapper to drift."""
        assert _rs._rs_alias_free_distance is _rs.rs_alias_free_distance
        from lumenairy.propagators.rs import (      # the importable form
            _rs_alias_free_distance as _priv,
            rs_alias_free_distance as _pub,
        )
        assert _priv is _pub

    @pytest.mark.parametrize('N, dx, lam', [
        (64, 2e-6, 632.8e-9),
        (128, 1e-6, 1.55e-6),
        (256, 0.5e-6, 400e-9),
        (32, 10e-6, 10.6e-6),
    ])
    def test_a20_value_is_the_closed_form(self, N, dx, lam):
        """Oracle: ``2*N*dx**2/lambda`` written out here.  Same three
        float64 operations on the same operands, so the error floor is
        zero and the bar is exact equality."""
        assert _rs.rs_alias_free_distance(N, dx, lam) == (
            2.0 * float(N) * float(dx) ** 2 / float(lam))

    def test_a20_is_the_kernel_auto_branch_point(self):
        """Why the name has to be public: it is the number
        ``rayleigh_sommerfeld_propagate`` routes on, so a caller choosing
        ``z`` / ``N`` / ``dx`` needs to be able to compute it.

        Pinned by behaviour, not by reading the source: just BELOW the
        distance, ``kernel='auto'`` is bit-identical to ``'transfer'`` and
        ``'spatial'`` REFUSES; at the distance, ``'auto'`` is bit-identical
        to ``'spatial'``.  Bit-identity (``array_equal``) because 'auto'
        does not compute anything of its own -- it selects a branch.
        """
        N, dx, lam = 64, 1e-6, 633e-9
        z_a = _rs.rs_alias_free_distance(N, dx, lam)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E = np.exp(-(X ** 2 + Y ** 2) / (6e-6) ** 2).astype(np.complex128)

        below = z_a * 0.5
        auto_lo = _rs.rayleigh_sommerfeld_propagate(E, below, lam, dx)
        tf_lo = _rs.rayleigh_sommerfeld_propagate(E, below, lam, dx,
                                                  kernel='transfer')
        assert np.array_equal(auto_lo, tf_lo)
        with pytest.raises(ValueError, match='2\\*N\\*dx\\*\\*2/wavelength'):
            _rs.rayleigh_sommerfeld_propagate(E, below, lam, dx,
                                              kernel='spatial')

        auto_hi = _rs.rayleigh_sommerfeld_propagate(E, z_a, lam, dx)
        sp_hi = _rs.rayleigh_sommerfeld_propagate(E, z_a, lam, dx,
                                                  kernel='spatial')
        assert np.array_equal(auto_hi, sp_hi)
