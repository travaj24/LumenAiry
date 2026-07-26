"""S12-2 -- ``coherence_dock``'s Tab 1 "Source shape" combo was inert.

Provenance
----------
``54a2dcf`` (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A) flagged
but did not claim: *"coherence_dock's source-shape combo has never been
wired (pre-existing inert control)"*.  Confirmed and fixed here.

The combo was constructed, populated with four entries (Circular /
Annular / Dipole / Quadrupole) and laid into the form -- but ``_run``
built its params dict without ever reading ``combo_shape``, so all four
entries produced the identical filled-disk image.  ``koehler_image``, the
only library call Tab 1 made, takes a scalar ``condenser_NA`` and can
describe nothing but a filled disk, so the combo had no route to the
physics even in principle.

Fix
---
1. ``build_source_angles(shape, half_angle_rad, n)`` -- a module-level,
   Qt-free helper that maps a shape name onto an explicit source-point set
   suitable for ``extended_source_image(source_angles=, source_weights=)``.
   Tab 1 and Tab 3 both call it (Tab 3's ``_build_source_angles``
   staticmethod is now a delegate), so there is one implementation and one
   set of pins.
2. ``_run`` passes ``shape``; ``_KoehlerWorker.run`` keeps
   ``koehler_image`` for 'Circular' (that IS the filled disk, so those
   numbers do not move) and routes the three non-disk pupils through
   ``extended_source_image``.

Geometry, as measured
---------------------
Cartesian-masked shapes (Circular / Annular / Gaussian / Custom) keep the
pre-fix construction bit-for-bit -- verified identical at n = 3, 9, 16, 33
for all four.  Pole shapes are built in POLAR coordinates because a
Cartesian mask cannot express them robustly: at hw=1 the annulus mask
admits 4 points at n=3, **0 at n=4**, 8 at n=5.  Point counts at hw=0.1:

    shape        n=3   n=4   n=9   n=16
    Circular       5     4    49    172
    Annular        4  RAISE   28    112
    Gaussian       9    16    81    256
    Custom         1     4    45    172
    Dipole         8     8    32    128
    Quadrupole    16    16    64    256

A second silent failure fell out of that measurement and is fixed too:
``extended_source_image`` accepts an empty ``source_angles`` and returns an
all-zero image without complaint (measured: shape preserved, sum 0.0, all
finite), so Tab 3 'Annular' at source_n=4 rendered a black frame with no
error anywhere.  ``build_source_angles`` now raises instead.

Headless method
---------------
PySide6 is absent on CI and on the audit box, so ``coherence_dock`` cannot
be imported.  The Qt-free module-level defs are lifted out with ``ast`` and
executed in a numpy-only namespace -- the pins therefore exercise the REAL
helper source, not a hand-copied duplicate, the same discipline
``test_niche_audit_w3_ui_deprecation.py`` uses.
"""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest

import lumenairy as la

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DOCK = os.path.join(REPO_ROOT, 'lumenairy', 'ui', 'coherence_dock.py')

HW = 0.1                      # source half-angle [rad] used throughout
POLE_HALF_ANGLE_DEG = 30.0    # wedge half-width the helper declares


def _dock_source() -> str:
    with open(DOCK, encoding='utf-8') as fh:
        return fh.read()


def _load_helper():
    """Exec the module-level, Qt-free source-geometry defs from the dock in
    an isolated numpy-only namespace."""
    want = {'build_source_angles', 'SOURCE_SHAPES',
            '_SOURCE_POLE_AZIMUTHS_DEG', '_SOURCE_POLE_HALF_ANGLE_DEG',
            '_SOURCE_INNER_FRAC'}
    tree = ast.parse(_dock_source(), filename=DOCK)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & want:
                keep.append(node)
    ns = {'np': np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), DOCK, 'exec'), ns)
    missing = want - set(ns)
    assert not missing, (
        f'coherence_dock no longer defines {sorted(missing)} at module '
        f'scope -- the shape mapping must stay Qt-free and importable so it '
        f'can be pinned without PySide6.')
    return ns


@pytest.fixture(scope='module')
def helper():
    return _load_helper()


@pytest.fixture(scope='module')
def bsa(helper):
    return helper['build_source_angles']


def _xy(angles):
    a = np.asarray(angles, dtype=float)
    assert a.ndim == 2 and a.shape[1] == 2, (
        f'angles must be a sequence of (ax, ay) pairs; got shape {a.shape}')
    return a


# ===========================================================================
# The library route the wiring depends on
# ===========================================================================

class TestLibraryRouteExists:

    def test_extended_source_image_takes_an_explicit_point_set(self):
        params = inspect.signature(la.extended_source_image).parameters
        for needed in ('object_field', 'prescription', 'wavelength', 'dx',
                       'source_angles', 'source_weights'):
            assert needed in params, (
                f'extended_source_image lost {needed!r}; the non-disk '
                f'source shapes have no other route to the physics.')

    def test_koehler_image_cannot_express_a_non_disk_pupil(self):
        """Discriminator: this is WHY the combo could not have worked while
        Tab 1 only called koehler_image."""
        params = inspect.signature(la.koehler_image).parameters
        assert 'condenser_NA' in params
        for absent in ('source_angles', 'source_weights', 'shape',
                       'source_shape'):
            assert absent not in params, (
                f'koehler_image gained {absent!r}: Tab 1 could route the '
                f'shape through it directly and this pin should be revisited')


# ===========================================================================
# The helper's four Tab-1 shapes
# ===========================================================================

class TestShapeGeometry:

    @pytest.mark.parametrize('shape', ['Circular', 'Annular', 'Gaussian',
                                       'Custom', 'Dipole', 'Quadrupole'])
    def test_contract_is_well_formed(self, bsa, shape):
        angles, weights = bsa(shape, HW, 9)
        assert len(angles) == len(weights) > 0
        assert all(w > 0 for w in weights), 'weights must be positive'
        a = _xy(angles)
        assert np.all(np.isfinite(a))

    # Point counts measured at hw=0.1, tabulated in the module docstring.
    @pytest.mark.parametrize('shape,n,count', [
        ('Circular', 3, 5), ('Circular', 9, 49), ('Circular', 16, 172),
        ('Annular', 3, 4), ('Annular', 9, 28), ('Annular', 16, 112),
        ('Gaussian', 9, 81), ('Gaussian', 16, 256),
        ('Custom', 9, 45), ('Custom', 16, 172),
        ('Dipole', 3, 8), ('Dipole', 4, 8), ('Dipole', 9, 32),
        ('Dipole', 16, 128),
        ('Quadrupole', 3, 16), ('Quadrupole', 4, 16), ('Quadrupole', 9, 64),
        ('Quadrupole', 16, 256),
    ])
    def test_point_counts(self, bsa, shape, n, count):
        angles, _ = bsa(shape, HW, n)
        assert len(angles) == count

    def test_pole_counts_follow_the_declared_formula(self, bsa, helper):
        """``n_poles * max(2, round(n / 2))**2``."""
        for shape, n_poles in (('Dipole', 2), ('Quadrupole', 4)):
            assert len(helper['_SOURCE_POLE_AZIMUTHS_DEG'][shape]) == n_poles
            for n in (3, 4, 5, 9, 16, 33):
                m = max(2, int(round(n / 2)))
                angles, _ = bsa(shape, HW, n)
                assert len(angles) == n_poles * m * m, (
                    f'{shape} n={n}: expected {n_poles}*{m}**2, '
                    f'got {len(angles)}')

    @pytest.mark.parametrize('shape', ['Circular', 'Annular', 'Gaussian',
                                       'Custom', 'Dipole', 'Quadrupole'])
    def test_distribution_is_centred(self, bsa, shape):
        """Every pupil is symmetric about the axis, so the weighted mean
        direction must be on-axis -- otherwise the shape would introduce a
        spurious image shift."""
        angles, weights = bsa(shape, HW, 9)
        a, w = _xy(angles), np.asarray(weights, dtype=float)
        centroid = (a * w[:, None]).sum(axis=0) / w.sum()
        assert np.allclose(centroid, 0.0, atol=1e-12 * HW + 1e-15), (
            f'{shape} centroid {centroid} is off-axis')

    def test_circular_fills_the_disk(self, bsa):
        a = _xy(bsa('Circular', HW, 17)[0])
        r = np.hypot(a[:, 0], a[:, 1])
        assert r.max() <= HW * (1 + 1e-12)
        assert r.min() == pytest.approx(0.0, abs=1e-15), (
            'an odd-n circular pupil must include the on-axis point')

    def test_annular_is_a_ring_with_a_hole(self, bsa, helper):
        inner = helper['_SOURCE_INNER_FRAC']
        a = _xy(bsa('Annular', HW, 17)[0])
        r = np.hypot(a[:, 0], a[:, 1])
        assert r.min() >= inner * HW * (1 - 1e-12), (
            f'annular pupil must be empty inside {inner} * hw; '
            f'min r/hw = {r.min() / HW:.4f}')
        assert r.max() <= HW * (1 + 1e-12)

    @pytest.mark.parametrize('shape,axes', [
        ('Dipole', (0.0, 180.0)),
        ('Quadrupole', (45.0, 135.0, 225.0, 315.0)),
    ])
    def test_poles_sit_in_wedges_about_their_axes(self, bsa, shape, axes):
        a = _xy(bsa(shape, HW, 9)[0])
        r = np.hypot(a[:, 0], a[:, 1])
        assert r.min() >= 0.6 * HW * (1 - 1e-12)
        assert r.max() <= HW * (1 + 1e-12)
        az = np.degrees(np.arctan2(a[:, 1], a[:, 0])) % 360.0
        dev = np.min(
            np.abs(((az[:, None] - np.asarray(axes)[None, :] + 180) % 360)
                   - 180), axis=1)
        assert dev.max() <= POLE_HALF_ANGLE_DEG + 1e-9, (
            f'{shape} has a point {dev.max():.4f} deg off its nearest pole '
            f'axis; the wedge half-angle is {POLE_HALF_ANGLE_DEG}')
        # every wedge must actually be populated
        nearest = np.argmin(
            np.abs(((az[:, None] - np.asarray(axes)[None, :] + 180) % 360)
                   - 180), axis=1)
        assert len(set(nearest.tolist())) == len(axes), (
            f'{shape} left one of its {len(axes)} poles empty')

    def test_dipole_is_point_symmetric(self, bsa):
        a = _xy(bsa('Dipole', HW, 9)[0])
        pts = {(round(x, 12), round(y, 12)) for x, y in a}
        assert all((round(-x, 12), round(-y, 12)) in pts for x, y in a), (
            'a dipole must be symmetric under p -> -p')

    def test_quadrupole_is_invariant_under_90_degree_rotation(self, bsa):
        a = _xy(bsa('Quadrupole', HW, 9)[0])
        pts = {(round(x, 12), round(y, 12)) for x, y in a}
        assert all((round(-y, 12), round(x, 12)) in pts for x, y in a), (
            'a quadrupole must be invariant under a 90 deg rotation')

    def test_the_four_tab1_shapes_are_all_distinct(self, bsa):
        """The point of the fix: picking a different entry must change the
        source.  Pre-fix all four produced the identical disk."""
        sets = {}
        for shape in ('Circular', 'Annular', 'Dipole', 'Quadrupole'):
            angles, _ = bsa(shape, HW, 9)
            sets[shape] = frozenset(
                (round(x, 12), round(y, 12)) for x, y in angles)
        names = sorted(sets)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                assert sets[first] != sets[second], (
                    f'{first} and {second} produce the same source-point '
                    f'set -- the combo is inert again')

    def test_scaling_the_half_angle_scales_the_pupil(self, bsa):
        for shape in ('Circular', 'Annular', 'Dipole', 'Quadrupole'):
            a1 = _xy(bsa(shape, HW, 9)[0])
            a2 = _xy(bsa(shape, 2 * HW, 9)[0])
            assert np.allclose(a2, 2 * a1), (
                f'{shape} must scale linearly with half_angle_rad')


# ===========================================================================
# Loud failure instead of silent degradation
# ===========================================================================

class TestFailsLoudly:

    def test_unknown_shape_raises(self, bsa, helper):
        with pytest.raises(ValueError, match='unknown source shape'):
            bsa('Bowtie', HW, 9)
        assert 'Bowtie' not in helper['SOURCE_SHAPES']

    def test_every_combo_entry_is_a_known_shape(self, helper):
        """Both combos' entries must be shapes the helper implements --
        this is the pin that would have caught the original bug class."""
        src = _dock_source()
        listed = set()
        for node in ast.walk(ast.parse(src, filename=DOCK)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', None) == 'addItems'
                    and node.args
                    and isinstance(node.args[0], ast.List)):
                for elt in node.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                            elt.value, str):
                        listed.add(elt.value)
        assert 'Dipole' in listed and 'Quadrupole' in listed, (
            f'expected the Tab 1 shape combo entries; found {sorted(listed)}')
        unknown = listed - set(helper['SOURCE_SHAPES'])
        assert not unknown, (
            f'combo entries {sorted(unknown)} have no distribution behind '
            f'them -- they would be inert controls')

    def test_empty_mask_raises_instead_of_a_black_frame(self, bsa):
        """Annular at n=4 admits zero Cartesian nodes (measured), and
        extended_source_image would silently return an all-zero image."""
        with pytest.raises(ValueError, match='admits no sample points'):
            bsa('Annular', HW, 4)

    def test_empty_source_angles_really_is_silent_in_the_library(self):
        """Discriminator for the pin above: the library does NOT complain,
        so the helper has to."""
        obj = np.ones((32, 32), dtype=complex)
        pres = {'surfaces': [{'type': 'standard', 'radius': np.inf,
                              'thickness': 1e-3, 'material': 'air'}]}
        out = np.asarray(la.extended_source_image(
            object_field=obj, prescription=pres, wavelength=550e-9,
            dx=1e-6, source_angles=[], source_weights=[]))
        assert out.shape == (32, 32)
        assert np.all(np.isfinite(out)) and float(np.nansum(out)) == 0.0, (
            'extended_source_image no longer returns a silent zero image '
            'for an empty source; the helper guard could be relaxed')

    def test_zero_half_angle_is_clamped_not_divided_by(self, bsa):
        for shape in ('Circular', 'Annular', 'Custom', 'Dipole',
                      'Quadrupole'):
            angles, weights = bsa(shape, 0.0, 9)
            assert len(angles) == len(weights) > 0
            assert np.all(np.isfinite(_xy(angles)))


# ===========================================================================
# Tab 3 must not move; Tab 1 must now read the combo
# ===========================================================================

class TestNoRegressionAndWiring:

    @pytest.mark.parametrize('shape', ['Circular', 'Annular', 'Gaussian',
                                       'Custom'])
    @pytest.mark.parametrize('n', [3, 9, 16, 33])
    def test_cartesian_shapes_are_bit_identical_to_the_pre_fix_helper(
            self, bsa, shape, n):
        """Tab 3 shipped these four; refactoring them into the shared
        helper must not move a single number."""
        hw = max(HW, 1e-9)
        ax = np.linspace(-hw, hw, n)
        AX, AY = np.meshgrid(ax, ax)
        R = np.sqrt(AX ** 2 + AY ** 2)
        if shape == 'Annular':
            W = ((R <= hw) & (R >= 0.6 * hw)).astype(np.float64)
        elif shape == 'Gaussian':
            W = np.exp(-(R ** 2) / (2.0 * (hw / 2.355) ** 2))
        elif shape == 'Custom':
            W = np.clip(1.0 - (R / hw), 0.0, 1.0)
        else:
            W = (R <= hw).astype(np.float64)
        mask = W > 0
        expect_a = list(zip(AX[mask].ravel(), AY[mask].ravel()))
        expect_w = W[mask].ravel().tolist()

        angles, weights = bsa(shape, HW, n)
        assert np.array_equal(np.asarray(angles), np.asarray(expect_a))
        assert np.array_equal(np.asarray(weights), np.asarray(expect_w))

    def test_tab1_run_puts_shape_into_params(self):
        """The one missing line.  ``_run`` must read ``combo_shape`` --
        FAILS on a pre-fix worktree."""
        src = _dock_source()
        assert 'shape=self.combo_shape.currentText()' in src, (
            "CoherenceDock._run does not pass the combo's value: the "
            'Source shape control is inert again (S12-2).')

    def test_tab1_worker_consumes_shape_and_branches(self):
        src = _dock_source()
        assert "self.params.get('shape'" in src, (
            '_KoehlerWorker.run never reads params["shape"]')
        assert 'build_source_angles(' in src
        assert 'extended_source_image(' in src, (
            'the non-disk shapes need extended_source_image; without it '
            'only Circular can be expressed')

    def test_tab3_delegates_to_the_shared_helper(self):
        """One implementation, one set of pins."""
        tree = ast.parse(_dock_source(), filename=DOCK)
        found = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == '_build_source_angles'):
                found = True
                calls = [n for n in ast.walk(node)
                         if isinstance(n, ast.Call)
                         and getattr(n.func, 'id', None)
                         == 'build_source_angles']
                assert calls, (
                    '_build_source_angles no longer delegates to the '
                    'module-level helper; Tab 1 and Tab 3 would drift')
        assert found, 'Tab 3 lost its _build_source_angles entry point'

    def test_koehler_image_still_has_two_call_sites(self):
        """Guards the pre-existing W3 pin (``n >= 2``): Tab 1 keeps its
        koehler_image call for the filled-disk case."""
        n = sum(
            1 for node in ast.walk(ast.parse(_dock_source(), filename=DOCK))
            if isinstance(node, ast.Call)
            and getattr(node.func, 'attr', None) == 'koehler_image')
        assert n >= 2, (
            f'expected the Schell tab and the Koehler tab call sites; '
            f'found {n}.  test_niche_audit_w3_ui_deprecation.py asserts '
            f'>= 2 and would fail too.')
