"""VERIFY-A9 — independent re-verification of WP-A9 (designer UI, U1-U7).

These tests are deliberately NOT the ones WP-A9 wrote: every fixture is a
fold topology or a sampling density the WP did not use, and every oracle is
built inside this file (a hand-written Welford paraxial matrix product, a
hand-written paraxial marginal-ray trace, a hand-built unfolded surface
list, a re-implementation of the PRE-fix semi-diameter rule) so that no
assertion is checked against the code under test.

Harness.  PySide6 is not installed on this workstation; the auditor's Qt
stub under ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/
stub`` is the harness, and it is installed, used and *parked* by
``test_audit2609_a9_ui``.  This file reuses that bootstrap rather than
standing up a second copy, so there is exactly one owner of the
install/park discipline that keeps the sibling UI test files' skip set
(38 skips, measured with and without this file) unchanged.  Nothing here
skips on PySide6's absence -- a missing Qt is a harness problem, not a
reason to stop testing the physics glue.

Error floors.  Every "two paths, same arithmetic" comparison below is
between two float64 evaluations that differ only in the ORDER of the same
multiplications, so the floor is ~1e-16 relative; those bars sit at 1e-12
(four decades of headroom) and the pre-fix values quoted in each docstring
are 10-12 decades on the wrong side.  Where the comparison is real-ray
against paraxial the bar is set from the aberration the fixture actually
exhibits, measured and quoted in the comment.
"""
import os
import sys
import types
import pathlib

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / 'tests' / 'unit') not in sys.path:
    sys.path.insert(0, str(_REPO / 'tests' / 'unit'))

import numpy as np                                   # noqa: E402
import pytest                                        # noqa: E402

import test_audit2609_a9_ui as _h                    # noqa: E402

UI = _h.UI
QT_FLAVOUR = _h.QT_FLAVOUR
model = UI['model']
SystemModel = model.SystemModel
SurfaceRow = model.SurfaceRow
Element = model.Element
SourceDefinition = model.SourceDefinition
WV = UI['waveoptics_dock']

from lumenairy.raytrace import (                     # noqa: E402
    Surface, surfaces_from_prescription, system_abcd, find_paraxial_focus,
)
from lumenairy.glass import get_glass_index          # noqa: E402


@pytest.fixture(autouse=True)
def _qt_stub_visible():
    """Un-park the shared Qt stub for the duration of each test."""
    if QT_FLAVOUR == 'real':
        yield
        return
    _h._unpark_stub()
    try:
        yield
    finally:
        _h._park_stub()


# ---------------------------------------------------------------------------
# Independent oracles (nothing below calls the code under test)
# ---------------------------------------------------------------------------

def _hand_abcd(surfs, wv):
    """Paraxial system matrix, written from Welford's rules here.

    Mirror: ``n2 = -n1`` with power ``(n2 - n1) / R`` and a parity flip on
    every following leg; coord break: a pure ``t / n`` transfer.  Returns
    ``(efl, bfl)`` with ``efl = -1/C`` and ``bfl = |n_image| * (-A/C)``,
    the conventions ``raytrace.seidel.system_abcd`` documents.  Shares no
    code with it.
    """
    M = np.eye(2)
    parity = 0
    n_end = 1.0
    for i, s in enumerate(surfs):
        sign = -1.0 if parity else 1.0
        if getattr(s, 'is_coordbrk', False):
            if i < len(surfs) - 1:
                n_after = sign * get_glass_index(s.glass_after, wv)
                M = np.array([[1.0, s.thickness / n_after],
                              [0.0, 1.0]]) @ M
            continue
        n1 = sign * get_glass_index(s.glass_before, wv)
        if s.is_mirror:
            n2 = -n1
            if np.isfinite(s.radius):
                M = np.array([[1.0, 0.0],
                              [-(n2 - n1) / s.radius, 1.0]]) @ M
            parity ^= 1
        else:
            n2 = sign * get_glass_index(s.glass_after, wv)
            if np.isfinite(s.radius):
                M = np.array([[1.0, 0.0],
                              [-(n2 - n1) / s.radius, 1.0]]) @ M
        if i < len(surfs) - 1:
            M = np.array([[1.0, s.thickness / n2], [0.0, 1.0]]) @ M
        n_end = abs(n2)
    A, C = M[0, 0], M[1, 0]
    return -1.0 / C, n_end * (-A / C)


def _hand_marginal_heights(surfs, y0, wv):
    """Paraxial marginal-ray height at each non-cb surface (y=y0, u=0)."""
    y, u, parity, heights = y0, 0.0, 0, []
    for s in surfs:
        if getattr(s, 'is_coordbrk', False):
            continue
        sign = -1.0 if parity else 1.0
        n1 = sign * get_glass_index(s.glass_before, wv)
        if s.is_mirror:
            n2 = -n1
            parity ^= 1
        else:
            n2 = sign * get_glass_index(s.glass_after, wv)
        heights.append(y)
        phi = (n2 - n1) / s.radius if np.isfinite(s.radius) else 0.0
        u = (n1 * u - phi * y) / n2
        y = y + s.thickness * u
    return heights


def _prefix_semi_diameters(pres):
    """The semi-diameter rule EXACTLY as ``trace.py`` shipped it before
    this verification pass: the per-surface key replaces the
    ``aperture_diameter / 2`` default and is then ``min()``-ed against the
    i-th ``element_type == 'surface'`` entry of ``elements``."""
    aperture = pres.get('aperture_diameter')
    elements = pres.get('elements', None)
    out = []
    for i, ps in enumerate(pres['surfaces']):
        sd = np.inf if aperture is None else aperture / 2.0
        ps_sd = ps.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        if elements is not None:
            refr = [e for e in elements
                    if e.get('element_type') == 'surface']
            if i < len(refr):
                esd = refr[i].get('semi_diameter', np.inf)
                if esd is not None and esd > 0 and np.isfinite(esd):
                    sd = min(sd, esd)
        out.append(float(sd))
    return out


# ---------------------------------------------------------------------------
# Fold topologies WP-A9 did not use
# ---------------------------------------------------------------------------

def _mirror(R, sd, **kw):
    return SurfaceRow(R, 0.0, '', sd, surf_type='Mirror', **kw)


def _two_mirror_model():
    """Two concave mirrors with alternating Zemax-signed gaps, then a
    singlet.  Two mirrors shift the refracting-only ``elements`` matcher
    by TWO, which is the case the audit's single-mirror fixture cannot
    exercise."""
    m = SystemModel()
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=100.0,
                                surfaces=[_mirror(-300.0, 25.0)]))
    m.insert_element(2, Element(0, 'M2', 'Mirror', distance_mm=-80.0,
                                surfaces=[_mirror(-200.0, 20.0)]))
    m.insert_element(3, Element(0, 'L1', 'Singlet', distance_mm=60.0,
                                surfaces=[SurfaceRow(70.0, 4.0, 'N-BK7', 8.0),
                                          SurfaceRow(np.inf, 0.0, '', 8.0)]))
    return m


def _mirror_then_cb_model():
    """A CURVED 45-degree fold mirror followed by a tilted biconvex
    singlet: the post-mirror air gap rides the coord break's thickness,
    the one place the legacy export used to drop it."""
    m = SystemModel()
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[_mirror(-200.0, 25.0)],
                                tilt_x=45.0))
    m.insert_element(2, Element(0, 'L1', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(60.0, 3.0, 'N-BK7', 10.0),
                                          SurfaceRow(-60.0, 0.0, '', 10.0)],
                                tilt_x=45.0))
    return m


def _mirror_last_model():
    """A singlet followed by a concave mirror as the LAST surface: the
    image space is post-reflection, so the mirror parity has to survive
    the export for BFL to come out right."""
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=15.0,
                                surfaces=[SurfaceRow(90.0, 5.0, 'N-BK7', 12.0),
                                          SurfaceRow(np.inf, 0.0, '', 12.0)]))
    m.insert_element(2, Element(0, 'M1', 'Mirror', distance_mm=45.0,
                                surfaces=[_mirror(-250.0, 30.0)]))
    return m


def _stop_on_mirror_model():
    m = SystemModel()
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[_mirror(-200.0, 25.0, is_stop=True)]))
    m.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=-30.0,
                                surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                                          SurfaceRow(np.inf, 0.0, '', 6.0)]))
    return m


# ---------------------------------------------------------------------------
# U1 -- the exported prescription is the system the layout draws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name,build', [
    ('two_mirrors', _two_mirror_model),
    ('mirror_then_coord_break', _mirror_then_cb_model),
    ('mirror_last', _mirror_last_model),
    ('stop_on_mirror', _stop_on_mirror_model),
])
def test_u1_folded_export_matches_an_independent_abcd(name, build):
    """Three ways of computing the same system must agree:

    1. ``system_abcd(model.build_trace_surfaces())`` -- the layout;
    2. ``system_abcd(surfaces_from_prescription(model.to_prescription()))``
       -- what the 18 dock call sites analyse;
    3. ``_hand_abcd`` above -- Welford's rules written out in this file.

    (2) vs (1) is the U1 property; (3) proves neither is echoing a shared
    mistake.  All three are float64 evaluations of the same product, so
    the bar is 1e-12 relative -- four decades above the ~1e-16 floor.
    Pre-fix, (2) differed from (1) by 11.1 % in EFL and 22.2 % in BFL on
    the audit's single-mirror fixture because ``is_mirror`` was stripped
    while the Zemax-signed negative gap survived.
    """
    m = build()
    wv = m.wavelength_m
    local = m.build_trace_surfaces()
    exported = surfaces_from_prescription(m.to_prescription())

    _, efl_layout, bfl_layout, _ = system_abcd(local, wv)
    _, efl_exp, bfl_exp, _ = system_abcd(exported, wv)
    efl_hand, bfl_hand = _hand_abcd(local, wv)

    assert efl_layout == pytest.approx(efl_hand, rel=1e-12), name
    assert bfl_layout == pytest.approx(bfl_hand, rel=1e-12), name
    assert efl_exp == pytest.approx(efl_layout, rel=1e-12), name
    assert bfl_exp == pytest.approx(bfl_layout, rel=1e-12), name

    # The mirror flags survive, and so does the unsigned axial path.
    assert [s.is_mirror for s in exported] == [
        s.is_mirror for s in local if not s.is_coordbrk], name
    assert sum(abs(s.thickness) for s in exported) == pytest.approx(
        sum(abs(s.thickness) for s in local), rel=1e-12), name


def test_u1_is_stop_on_a_mirror_round_trips():
    """The stop can sit on the fold.  Pre-fix ``SurfaceRow`` had no
    ``is_stop`` at all, so the flag could not be held, exported or
    re-imported."""
    m = _stop_on_mirror_model()
    pres = m.to_prescription()
    assert [s['is_stop'] for s in pres['surfaces']] == [True, False, False]
    assert pres['stop_index'] == 0
    assert [s.is_stop for s in surfaces_from_prescription(pres)] == \
        [True, False, False]

    m2 = SystemModel()
    m2.load_prescription(pres)
    stops = [(e.elem_type, s.is_stop) for e in m2.elements for s in e.surfaces]
    assert [t for t, flag in stops if flag] == ['Mirror'], stops


# ---------------------------------------------------------------------------
# U2 -- per-surface semi-diameters through export, load and trace
# ---------------------------------------------------------------------------

def test_u2_per_surface_semi_diameter_is_not_clipped_by_the_elements_matcher():
    """``trace.py``'s ``elements`` matcher used to ``min()`` against a
    PRESENT per-surface key with a refracting-only index, so each mirror
    shifted the mapping by one and a fold was clamped to the following
    lens's aperture.

    Measured on these two fixtures with the shipped rule
    (``_prefix_semi_diameters``, replayed in-test): one mirror
    ``[0.006, 0.006, 0.006]``, two mirrors
    ``[0.008, 0.008, 0.008, 0.008]``.  Every entry must now be the
    semi-diameter the user typed, to the last bit.
    """
    m1 = SystemModel()
    m1.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                 surfaces=[_mirror(-200.0, 25.0)]))
    m1.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=-30.0,
                                 surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                                           SurfaceRow(np.inf, 0.0, '', 6.0)]))
    p1 = m1.to_prescription()
    got1 = [s.semi_diameter for s in surfaces_from_prescription(p1)]
    assert got1 == pytest.approx([0.025, 0.006, 0.006], rel=1e-15)
    assert _prefix_semi_diameters(p1) == pytest.approx(
        [0.006, 0.006, 0.006], rel=1e-15), 'pre-fix replay drifted'

    p2 = _two_mirror_model().to_prescription()
    got2 = [s.semi_diameter for s in surfaces_from_prescription(p2)]
    assert got2 == pytest.approx([0.025, 0.020, 0.008, 0.008], rel=1e-15)
    assert _prefix_semi_diameters(p2) == pytest.approx(
        [0.008, 0.008, 0.008, 0.008], rel=1e-15)


def test_u2_both_elements_shapes_index_correctly_without_a_per_surface_key():
    """The ``elements`` fallback has to serve two incompatible layouts.

    * lens-only ``surfaces`` (what ``load_zemax_zmx`` emits): mirrors are
      in ``elements`` but NOT in ``surfaces``, so surface i is the i-th
      ``element_type == 'surface'`` entry;
    * chronological ``surfaces`` (what the designer UI emits, and what a
      PRE-WP-A9 lens-library entry holds): ``elements`` is one entry per
      surface, so surface i is ``elements[i]``.

    Reading the first shape positionally hands the first lens surface the
    mirror's aperture; reading the second with the refracting-only filter
    hands the mirror the following lens's.  Both are asserted here.
    """
    def _flat(R, gb, ga, **kw):
        d = {'radius': R, 'conic': 0.0, 'glass_before': gb,
             'glass_after': ga}
        d.update(kw)
        return d

    lens_only = {
        'aperture_diameter': 0.0254,
        'surfaces': [_flat(0.08, 'air', 'N-BK7'),
                     _flat(np.inf, 'N-BK7', 'air')],
        'thicknesses': [0.004],
        'elements': [
            {'element_type': 'mirror', 'radius': -0.2,
             'semi_diameter': 0.025, 'surf_num': 0},
            {'element_type': 'surface', 'radius': 0.08,
             'semi_diameter': 0.006, 'surf_num': 1,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'element_type': 'surface', 'radius': np.inf,
             'semi_diameter': 0.006, 'surf_num': 2,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
    }
    assert [s.semi_diameter
            for s in surfaces_from_prescription(lens_only)] == \
        pytest.approx([0.006, 0.006], rel=1e-15)

    chronological = dict(lens_only)
    chronological['surfaces'] = [
        _flat(-0.2, 'air', 'air', is_mirror=True),
        _flat(0.08, 'air', 'N-BK7'),
        _flat(np.inf, 'N-BK7', 'air'),
    ]
    chronological['thicknesses'] = [-0.03, 0.004]
    got = [s.semi_diameter for s in surfaces_from_prescription(chronological)]
    # 0.0127 is ``aperture_diameter / 2``: the mirror's own 0.025 entry is
    # now the one consulted, and the documented ``min()`` against the
    # system aperture caps it there.  The point of the assertion is that
    # it is NOT 0.006 -- the FOLLOWING lens's aperture, which the
    # refracting-only index used to hand it.
    assert got == pytest.approx([0.0127, 0.006, 0.006], rel=1e-15)
    # Pre-fix: the mirror took the first LENS entry (0.006) and the last
    # lens surface ran off the end of the refracting-only list and fell
    # through to aperture_diameter / 2.
    assert _prefix_semi_diameters(chronological) == pytest.approx(
        [0.006, 0.006, 0.0127], rel=1e-15)


def test_u2_elements_fallback_is_bit_identical_for_every_library_producer():
    """Bit-identity guard on the ``trace.py`` change.

    The rule only moves for a prescription whose ``surfaces`` list
    carries mirrors -- i.e. the designer UI's.  Every builder and loader
    in the library must resolve exactly as it did before, so each is
    compared against ``_prefix_semi_diameters`` (the shipped rule
    replayed here).  Measured across the repository's 62 reachable
    prescriptions on 2026-09-12: 58 identical, 4 changed, and all four
    are UI folded exports.
    """
    import glob
    import warnings
    import lumenairy as la
    from lumenairy.io import normalize_prescription

    cases = {
        'make_singlet': la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                                        glass='N-BK7', aperture=25.4e-3),
        'make_doublet': la.make_doublet(R1=60e-3, R2=-40e-3, R3=-120e-3,
                                        d1=6e-3, d2=3e-3, glass1='N-BK7',
                                        glass2='N-SF2', aperture=25.4e-3),
        'make_biconic': la.make_biconic(R1_x=50e-3, R1_y=60e-3,
                                        R2_x=np.inf, R2_y=np.inf, d=3e-3,
                                        glass='N-BK7', aperture=25.4e-3),
    }
    repro = (_REPO / 'docs' / 'audits'
             / 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11' / 'repro')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for path in sorted(glob.glob(str(repro / '**' / '*.zmx'),
                                     recursive=True)):
            try:
                cases['zmx:' + os.path.basename(path)] = \
                    la.load_zemax_zmx(path)
            except Exception:
                continue        # malformed-by-design fixtures
        for path in sorted(glob.glob(str(repro / '**' / '*.seq'),
                                     recursive=True)):
            try:
                cases['seq:' + os.path.basename(path)] = \
                    la.load_codev_seq(path)
            except Exception:
                continue
        for key in list(cases):
            try:
                cases['norm:' + key] = normalize_prescription(cases[key])
            except Exception:
                continue

        assert len(cases) >= 20, f'only {len(cases)} prescriptions found'
        n_with_mirror = 0
        for key, pres in cases.items():
            if any(e.get('element_type') == 'mirror'
                   for e in (pres.get('elements') or ())):
                n_with_mirror += 1
            got = [s.semi_diameter for s in surfaces_from_prescription(pres)]
            want = _prefix_semi_diameters(pres)
            assert got == pytest.approx(want, rel=0.0, abs=0.0, nan_ok=True), \
                f'{key}: {got} != pre-fix {want}'
        # At least one of them contains a mirror, or the guard is vacuous.
        assert n_with_mirror >= 1


# ---------------------------------------------------------------------------
# U3 -- the point-source cone
# ---------------------------------------------------------------------------

def _capture_bundle(m, num_rings, rays_per_ring):
    import lumenairy.raytrace as rt
    cap = {}
    orig = rt._make_bundle

    def spy(x, y, L, M, wv):
        cap['L'] = np.asarray(L, dtype=float)
        cap['M'] = np.asarray(M, dtype=float)
        return orig(x, y, L, M, wv)

    rt._make_bundle = spy
    try:
        m.run_trace(num_rings=num_rings, rays_per_ring=rays_per_ring)
    finally:
        rt._make_bundle = orig
    return cap['L'], cap['M']


def _point_source_model(first_optic_mm, form_mm=1000.0):
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet',
                                distance_mm=first_optic_mm,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.epd_mm = 25.4
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=form_mm)
    m._invalidate()
    return m


@pytest.mark.parametrize('num_rings,rays_per_ring',
                         [(1, 1), (1, 3), (2, 5), (4, 12), (5, 1), (6, 7)])
def test_u3_cone_matches_the_closed_form_at_several_densities(num_rings,
                                                              rays_per_ring):
    """Closed form: ring k reaches ``rho_k = (k/num_rings) * semi_ap /
    obj_dist`` at azimuths ``2*pi*j/rays_per_ring``, plus one chief ray.

    Both sides are float64 evaluations of the same product/trig, so the
    bar is 1e-12 (four decades above the ~2e-16 the measurement actually
    shows).  Pre-fix every ray had ``L = M = rho``: ``num_rings``
    distinct directions instead of ``num_rings * rays_per_ring``, and a
    marginal ray sqrt(2) = 41 % too steep.
    """
    m = _point_source_model(200.0, form_mm=200.0)
    L, M = _capture_bundle(m, num_rings, rays_per_ring)

    assert L.size == num_rings * rays_per_ring + 1
    uniq = {(round(a, 12), round(b, 12)) for a, b in zip(L, M)}
    assert len(uniq) == L.size, f'degenerate bundle: {len(uniq)}/{L.size}'

    semi_ap = m.epd_m / 2.0
    obj = m.object_distance_m()
    rho = np.hypot(L, M)
    for k in range(1, num_rings + 1):
        sl = slice((k - 1) * rays_per_ring, k * rays_per_ring)
        np.testing.assert_allclose(rho[sl],
                                   (k / num_rings) * semi_ap / obj,
                                   rtol=1e-12)
        az = np.sort(np.mod(np.arctan2(M[sl], L[sl]), 2 * np.pi))
        np.testing.assert_allclose(
            az, np.arange(rays_per_ring) * 2 * np.pi / rays_per_ring,
            atol=1e-12)
    assert rho.max() == pytest.approx(semi_ap / obj, rel=1e-12)
    assert rho[-1] == pytest.approx(0.0, abs=1e-15)
    # Two-sided: the pre-fix marginal ray was sqrt(2) times this.
    assert rho.max() < np.sqrt(2) * semi_ap / obj * 0.99


def test_u3_object_distance_edges():
    """The geometric gap wins over the advisory form field; a first optic
    at a non-positive distance (a Zemax-signed post-mirror gap, or an
    element still at 0) falls back to the field rather than dividing by
    zero; a non-diverging source reports 0.0 = "object at infinity"."""
    assert _point_source_model(100.0, 1000.0).object_distance_m() == \
        pytest.approx(0.100, rel=1e-12)
    assert _point_source_model(0.0, 250.0).object_distance_m() == \
        pytest.approx(0.250, rel=1e-12)
    assert _point_source_model(-30.0, 250.0).object_distance_m() == \
        pytest.approx(0.250, rel=1e-12)

    # A very short conjugate must still produce a finite, on-shell bundle.
    m = _point_source_model(1e-3, 1000.0)
    assert m.object_distance_m() == pytest.approx(1e-6, rel=1e-12)
    L, M = _capture_bundle(m, 2, 4)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(M))

    for kind in ('plane_wave', 'gaussian', 'gaussian_aperture', 'top_hat',
                 'fiber_mode', 'emitter_array'):
        m = _point_source_model(100.0)
        m.elements[0].source = SourceDefinition(kind)
        m._invalidate()
        assert m.object_distance_m() == 0.0, kind
        assert m.to_prescription()['object_distance'] == 0.0, kind


# ---------------------------------------------------------------------------
# U4 -- process-global overrides
# ---------------------------------------------------------------------------

def _globals_state():
    from lumenairy.propagators import fft_infra
    from lumenairy.memory import get_max_ram
    return (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT, get_max_ram())


def test_u4_context_manager_restores_on_every_exit_including_baseexception():
    """``try/finally`` has to survive more than a plain ``Exception``: a
    ``KeyboardInterrupt`` through a generator-based context manager is the
    classic leak.  Nesting and a pre-existing user cap are checked too --
    the dock is not the only writer of these globals."""
    from lumenairy.memory import set_max_ram
    base = _globals_state()

    for backend in ('default', 'numpy', 'scipy', 'pyfftw', '', None):
        with WV._process_overrides(backend, None):
            pass
        assert _globals_state() == base, backend

    with pytest.raises(KeyboardInterrupt):
        with WV._process_overrides('scipy', 4):
            raise KeyboardInterrupt
    assert _globals_state() == base

    with WV._process_overrides('numpy', 4):
        mid = _globals_state()
        with WV._process_overrides('scipy', 8):
            assert _globals_state() != mid
        assert _globals_state() == mid
    assert _globals_state() == base

    set_max_ram(32)
    user = _globals_state()
    try:
        with WV._process_overrides('numpy', 4):
            pass
        assert _globals_state() == user
    finally:
        set_max_ram(None)
    assert _globals_state() == base


def _wave_worker(cfg_over=None, model_build=None):
    m = model_build() if model_build else _wave_singlet()
    cfg = dict(N=64, dx_m=40e-6, method='asm', backend='numpy',
               memory_limit_gb=4, lens_model='asm', save_planes={})
    cfg.update(cfg_over or {})
    w = WV.WaveOpticsWorker(m, cfg)
    got = []
    w.finished_result.connect(got.append)
    return w, got


def _wave_singlet():
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.epd_mm = 5.0
    m._invalidate()
    return m


def test_u4_the_real_worker_restores_the_globals_on_every_exit_path():
    """Success, cancellation and two flavours of failure, driven through
    ``WaveOpticsWorker.run()`` itself.  Pre-fix a default Run left
    ``USE_PYFFTW = USE_SCIPY_FFT = False`` for the life of the process."""
    base = _globals_state()

    w, got = _wave_worker()
    w.run()
    assert _globals_state() == base and len(got) == 1 and 'error' not in got[0]

    w, got = _wave_worker()
    w.requestInterruption()
    w.run()
    assert _globals_state() == base
    assert got == [{'error': 'Stopped by user'}]

    w, got = _wave_worker()

    def _boom():
        raise RuntimeError('injected')
    w._run_impl = _boom
    w.run()
    assert _globals_state() == base
    assert len(got) == 1 and 'injected' in got[0]['error']

    class _BadStr(Exception):
        def __str__(self):
            raise ValueError('broken __str__')

    w, got = _wave_worker()

    def _evil():
        raise _BadStr()
    w._run_impl = _evil
    w.run()          # must not escape run()
    assert _globals_state() == base
    assert len(got) == 1 and 'error' in got[0]

    # Combo index 0 ("Library default") touches nothing at all.
    w, got = _wave_worker({'backend': 'default', 'memory_limit_gb': None})
    w.run()
    assert _globals_state() == base and len(got) == 1


# ---------------------------------------------------------------------------
# U5 -- unfolding: WHERE the dropped mirror's gap lands
# ---------------------------------------------------------------------------

def _mirror_between_singlets():
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'M1', 'Mirror', distance_mm=25.0,
                                surfaces=[_mirror(np.inf, 25.0)]))
    m.insert_element(3, Element(0, 'L2', 'Singlet', distance_mm=-40.0,
                                surfaces=[SurfaceRow(80.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    return m


def test_u5_dropped_mirror_gap_lands_on_the_preceding_surface():
    """A ``Surface.thickness`` is the gap AFTER that surface, walked in
    the medium AFTER it (``_run_impl``'s per-surface loop).  Removing the
    mirror therefore merges the two legs that met at it onto the
    PRECEDING surface.

    Oracle: a hand-built unfolded list (below), and ``system_abcd`` on it.
    Carrying the gap forward instead put 40 mm of fold air inside the
    second lens's 3 mm N-BK7 leg: thicknesses
    ``[0.003, 0.025, 0.043, 0.0]`` and EFL/BFL 0.06823621185021057 /
    0.021085332140327456 m against the truth's 0.08251493722378367 /
    0.02484249389527306 m (17.3 % / 15.1 %).  Both sides of the bar are
    float64 evaluations of the same product, so 1e-12 relative is four
    decades above the floor.
    """
    m = _mirror_between_singlets()
    wv = m.wavelength_m
    folded = m.build_trace_surfaces()
    unfolded = WV._filter_wave_optics_surfaces(folded, unfold_mirrors=True,
                                               ignore_lateral_cbs=True)

    truth = [
        Surface(radius=0.05, thickness=0.003, glass_before='air',
                glass_after='N-BK7'),
        Surface(radius=np.inf, thickness=0.025 + 0.040, glass_before='N-BK7',
                glass_after='air'),
        Surface(radius=0.08, thickness=0.003, glass_before='air',
                glass_after='N-BK7'),
        Surface(radius=np.inf, thickness=0.0, glass_before='N-BK7',
                glass_after='air'),
    ]
    assert not any(s.is_mirror for s in unfolded)
    assert [float(s.thickness) for s in unfolded] == pytest.approx(
        [float(s.thickness) for s in truth], rel=1e-12, abs=1e-15)
    assert [s.glass_after for s in unfolded] == \
        [s.glass_after for s in truth]

    _, efl_got, bfl_got, _ = system_abcd(unfolded, wv)
    _, efl_want, bfl_want, _ = system_abcd(truth, wv)
    assert efl_got == pytest.approx(efl_want, rel=1e-12)
    assert bfl_got == pytest.approx(bfl_want, rel=1e-12)
    assert efl_got == pytest.approx(*(0.08251493722378367,), rel=1e-12)
    # Two-sided: nowhere near the forward-carry value.
    assert abs(efl_got - 0.06823621185021057) > 1e-3

    # Total unsigned axial path is still conserved when the fold is not
    # the leading surface.
    assert sum(s.thickness for s in unfolded) == pytest.approx(
        sum(abs(s.thickness) for s in folded), rel=1e-12)


def test_u5_a_leading_fold_drops_its_gap_and_a_trailing_fold_keeps_it():
    """The field is constructed AT the first surface, so a gap that
    precedes every kept surface is not propagated -- exactly as the
    source-to-first-surface gap already is not.  A gap that follows the
    last kept surface is real and must survive onto it."""
    lead = SystemModel()
    lead.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                   surfaces=[_mirror(-200.0, 25.0)]))
    lead.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=-30.0,
                                   surfaces=[SurfaceRow(80.0, 4.0, 'N-BK7',
                                                        6.0),
                                             SurfaceRow(np.inf, 0.0, '',
                                                        6.0)]))
    got = WV._filter_wave_optics_surfaces(lead.build_trace_surfaces(),
                                          unfold_mirrors=True,
                                          ignore_lateral_cbs=True)
    assert [float(s.thickness) for s in got] == pytest.approx([0.004, 0.0],
                                                              abs=1e-15)

    trail = SystemModel()
    trail.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=15.0,
                                    surfaces=[SurfaceRow(90.0, 5.0, 'N-BK7',
                                                         12.0),
                                              SurfaceRow(np.inf, 0.0, '',
                                                         12.0)]))
    trail.insert_element(2, Element(0, 'M1', 'Mirror', distance_mm=45.0,
                                    surfaces=[_mirror(-250.0, 30.0)]))
    trail.insert_element(3, Element(0, 'L3', 'Singlet', distance_mm=-20.0,
                                    surfaces=[SurfaceRow(70.0, 2.0, 'N-BK7',
                                                         10.0),
                                              SurfaceRow(np.inf, 0.0, '',
                                                         10.0)]))
    got = WV._filter_wave_optics_surfaces(trail.build_trace_surfaces(),
                                          unfold_mirrors=True,
                                          ignore_lateral_cbs=True)
    # 45 mm (L1 S2 -> mirror) + 20 mm (mirror -> L3 S1) merge onto L1 S2.
    assert [float(s.thickness) for s in got] == pytest.approx(
        [0.005, 0.065, 0.002, 0.0], abs=1e-15)


def test_u5_unfolded_equivalent_prescription_keeps_the_coord_break_gap():
    """``_prescription_from_surfaces`` folds coord-break Surfaces out of
    the list it hands the lens-model router.  Their TRANSFER thickness is
    real axial distance (a post-mirror gap rides it), so dropping the
    Surface must not drop the gap -- the U1 defect, one level down."""
    surfs = [
        Surface(radius=0.05, thickness=0.003, glass_before='air',
                glass_after='N-BK7', label='S1'),
        Surface(is_coordbrk=True, thickness=0.02, tilt_x_deg=30.0,
                glass_before='air', glass_after='air', label='CB'),
        Surface(radius=np.inf, thickness=0.004, glass_before='N-BK7',
                glass_after='air', label='S2'),
        Surface(radius=0.08, thickness=0.0, glass_before='air',
                glass_after='N-BK7', label='S3'),
    ]
    pres = WV._prescription_from_surfaces(surfs, 0.0254)
    assert pres['thicknesses'] == pytest.approx([0.003 + 0.02, 0.004],
                                                rel=1e-12)
    assert len(pres['surfaces']) == 3
    assert all(not s['is_mirror'] for s in pres['surfaces'])

    # A 45-degree fold end to end: filter, then export, then check the
    # total axial path against the folded list's.
    m = SystemModel()
    m.insert_element(1, Element(0, 'L0', 'Singlet', distance_mm=20.0,
                                surfaces=[SurfaceRow(120.0, 3.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[_mirror(np.inf, 25.0)],
                                tilt_x=45.0))
    m.insert_element(3, Element(0, 'L1', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)],
                                tilt_x=45.0))
    folded = m.build_trace_surfaces()
    unfolded = WV._filter_wave_optics_surfaces(folded, unfold_mirrors=True,
                                               ignore_lateral_cbs=True)
    assert [float(s.thickness) for s in unfolded] == pytest.approx(
        [0.003, 0.05 + 0.04, 0.003, 0.0], abs=1e-15)
    pres = WV._prescription_from_surfaces(unfolded, m.epd_m)
    assert pres['thicknesses'] == pytest.approx([0.003, 0.09, 0.003],
                                                rel=1e-12)


def test_u5_router_reports_a_non_fold_refusal_too():
    """The router's fallback line must carry the real reason, not just
    "folded".  A bad per-function kwarg makes ``apply_real_lens`` raise a
    ``TypeError``, which used to be swallowed into a thin-screen run
    labelled with the requested model."""
    w, got = _wave_worker({'lens_model': 'real_lens'})
    w._snap['lens_options'] = {'apply_real_lens': {'not_a_kwarg': 1}}
    w.run()
    r = got[-1]
    assert r['lens_model_requested'] == 'real_lens'
    assert r['lens_model_used'] == 'asm (fallback)'
    assert 'not_a_kwarg' in r['lens_model_fallback_reason']


# ---------------------------------------------------------------------------
# U6g -- the PSF/MTF exit pupil
# ---------------------------------------------------------------------------

def test_u6g_exit_pupil_radius_matches_an_independent_paraxial_marginal_ray():
    """On a two-element system the exit-pupil ray height is NOT ``EPD/2``
    (it is 5.82 mm for a 12.7 mm entrance semi-aperture here), so the
    oracle has to be a real marginal-ray trace, written out in
    ``_hand_marginal_heights``.

    Bar: 3 % relative.  The residual is the fixture's own spherical
    aberration -- real marginal rays land above the paraxial height --
    measured at 0.71 % (two elements) and 1.10 % (singlet) on
    2026-09-12; the defect being excluded is two DECADES away (the
    image-plane radius this code used to bin is 1.73e-04 m against a
    5.82e-03 m pupil, a factor of 34).
    """
    PSFMTFDock = UI['psf_mtf_dock'].PSFMTFDock
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(60.0, 4.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=60.0,
                                surfaces=[SurfaceRow(-200.0, 3.0, 'N-SF2',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.elements[-1].distance_mm = 0.0
    m._invalidate()
    result = m.run_trace(num_rings=8, rays_per_ring=36)

    rays, label = PSFMTFDock._exit_pupil_rays(result)
    assert rays is not None and label != 'Image'
    alive = rays.alive
    r_pup = float(np.max(np.hypot(rays.x[alive], rays.y[alive])))

    want = _hand_marginal_heights(m.build_trace_surfaces(), m.epd_m / 2.0,
                                  m.wavelength_m)[-1]
    assert r_pup == pytest.approx(want, rel=0.03)
    assert want != pytest.approx(m.epd_m / 2.0, rel=0.1), (
        'fixture no longer has a non-trivial exit pupil; oracle is vacuous')

    img = result.image_rays
    r_img = float(np.max(np.hypot(img.x[img.alive], img.y[img.alive])))
    assert r_pup > 10 * r_img
    # Exit-vertex referenced: every alive ray sits on ONE plane, so the
    # OPD is a wavefront and not each ray's own sag point.
    assert np.allclose(rays.z[alive], rays.z[alive][0], atol=1e-15)


# ---------------------------------------------------------------------------
# U7 -- the global search worker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('variables', [
    [(1, 0, 'radius')],
    [(1, 0, 'conic')],
    [(1, 0, 'radius'), (1, 0, 'conic')],
])
def test_u7_global_search_completes_and_emits_exactly_once(variables):
    """``opt_variables`` entries are ``(elem_idx, surf_idx, field)``
    triples; the restart loop unpacked them into two names, so
    ``GlobalSearchWorker.run()`` raised ``ValueError: too many values to
    unpack (expected 2, got 3)`` on the FIRST restart -- before any
    ``finished_result`` emission, leaving the dock's buttons disabled.
    """
    O = UI['optimizer_dock']
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.epd_mm = 25.4
    m.opt_variables = list(variables)
    m._invalidate()
    r0 = m.elements[1].surfaces[0].radius
    c0 = m.elements[1].surfaces[0].conic

    w = O.GlobalSearchWorker(m, 5, 3)
    got = []
    w.finished_result.connect(lambda ok, msg: got.append((ok, msg)))
    w.run()

    assert len(got) == 1 and got[0][0] is True, got
    assert w.result_x is not None and len(w.result_x) == len(variables)
    # The live model is never touched from the worker.
    assert m.elements[1].surfaces[0].radius == r0
    assert m.elements[1].surfaces[0].conic == c0


def test_u7_global_search_emits_once_even_when_the_merit_always_raises():
    O = UI['optimizer_dock']
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.opt_variables = [(1, 0, 'radius')]
    m._invalidate()
    w = O.GlobalSearchWorker(m, 5, 2)

    def _boom(x):
        raise RuntimeError('merit exploded')
    w.model.merit_function = _boom
    got = []
    w.finished_result.connect(lambda ok, msg: got.append((ok, msg)))
    w.run()
    assert len(got) == 1 and got[0][0] is False
    assert 'merit exploded' in got[0][1]


@pytest.mark.parametrize('mirror_tilt_deg', [0.0, 10.0, 20.0, 30.0, 45.0,
                                             60.0])
def test_u7_absolute_z_column_round_trips_through_a_fold(mirror_tilt_deg):
    """The absolute-coordinates column shows ``Element.origin[2]`` -- a
    world Z -- while the Distance field it writes back is measured along
    the optical axis.  On an unfolded system the two coincide; on a fold
    they differ by the axis's z-component, so typing the displayed value
    back MOVED the element (measured on a 30 deg fold: 40 mm -> 23.094
    mm; on a 45 deg fold: 40 mm -> 0 mm).

    Bar: 1e-9 mm absolute, which is nine decades below the smallest
    number the column can display (0.0001 mm) and eight above the
    float64 round-off of a ~50 mm coordinate (~7e-15 mm).  The 45 deg
    arm is the degenerate one -- the leg runs perpendicular to world Z,
    the column carries no information about the distance at all, and the
    write must be a no-op rather than a move.
    """
    m = SystemModel()
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[_mirror(np.inf, 25.0)],
                                tilt_x=mirror_tilt_deg))
    m.insert_element(2, Element(0, 'L1', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)],
                                tilt_x=mirror_tilt_deg))
    m.set_coordinate_mode('absolute')
    for idx in (1, 2):
        shown = m.get_display_distance(idx)
        assert shown == pytest.approx(
            float(np.asarray(m.elements[idx].origin, dtype=float)[2]),
            rel=1e-12), 'column no longer agrees with Element.origin'
        before = m.elements[idx].distance_mm
        m.set_display_distance(idx, shown)
        assert m.elements[idx].distance_mm == pytest.approx(before,
                                                            abs=1e-9)

    # A write the column cannot invert must not even take an undo step.
    if abs(m._display_distance_slope(2)) < 1e-9:
        depth = len(m._undo_stack)
        m.set_display_distance(2, m.get_display_distance(2) + 5.0)
        assert len(m._undo_stack) == depth
        assert m.elements[2].distance_mm == pytest.approx(40.0)


def test_u7_absolute_z_column_is_bit_identical_on_unfolded_systems():
    """Guard on the fold fix: an untilted leg has ``R[2, 2] == 1.0``
    exactly, so the division is the identity and the round trip is
    bit-exact, not merely close."""
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(80.0, 5.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.set_coordinate_mode('absolute')
    for idx in (1, 2):
        shown = m.get_display_distance(idx)
        before = m.elements[idx].distance_mm
        m.set_display_distance(idx, shown)
        assert m.elements[idx].distance_mm == before, 'not bit-exact'
    # 53.0 = 10 (gap) + 3 (L1 glass) + 40 (gap): the column carries the
    # internal thickness, which the pre-audit cumulative sum omitted.
    assert m.get_display_distance(2) == pytest.approx(53.0, rel=1e-12)


def test_u7_min_thickness_merit_counts_a_cemented_interface():
    """``surfaces[:-1]`` must still reach the INTERNAL surfaces of a
    multi-surface element: a cemented doublet's 0.4 mm cement leg is a
    real min-thickness violation and has to score ``(1 - 0.4)**2``, while
    the trailing surface (thickness 0 by the model's convention, its air
    gap living on ``Element.distance_mm``) must not."""
    m = SystemModel()
    m.insert_element(1, Element(0, 'D1', 'Doublet', distance_mm=20.0,
                                surfaces=[SurfaceRow(50.0, 5.0, 'N-BK7', 12.7),
                                          SurfaceRow(-40.0, 0.4, 'N-SF2',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.geo_merit_type = 'min_thickness'
    m.opt_variables = [(1, 0, 'radius')]
    m._invalidate()
    x = m.get_variable_values()
    assert m.merit_function(x) == pytest.approx((1.0 - 0.4) ** 2, rel=1e-12)
    m.elements[1].surfaces[1].thickness = 2.0
    assert m.merit_function(x) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Follow-up — the open items VERIFY-A9 raised, now closed
# ---------------------------------------------------------------------------

def test_followup_81_source_preset_rejects_an_unlisted_kind():
    """Every `source_type` branch in `to_source` / `describe` / the
    layout glyphs dispatches on membership of `SourceDefinition.TYPES`
    (which is also what the source-type combo is built from), so an
    unlisted string is not a source: it falls through every branch to
    the plane-wave default.  It used to be installed verbatim."""
    MainWindow = UI['main_window'].MainWindow
    m = SystemModel()
    m.set_wavelength(632.8)
    fake = types.SimpleNamespace(
        model=m, status_label=types.SimpleNamespace(setText=lambda t: None))

    with pytest.raises(ValueError, match='_ins_source_preset'):
        MainWindow._ins_source_preset(fake, 'not_a_kind')
    assert m.source.source_type == 'plane_wave', 'a bad preset mutated it'
    # Every listed kind is still accepted, and the six wired ones keep
    # their preset values.
    for kind in SourceDefinition.TYPES:
        MainWindow._ins_source_preset(fake, kind)
        assert m.source.source_type == kind
    MainWindow._ins_source_preset(fake, 'point_source')
    assert m.source.object_distance_mm == pytest.approx(1000.0)


def test_followup_83_emitter_counts_are_capped():
    """`to_source` loops over `emitter_nx * emitter_ny` emitters in
    Python, so an unbounded count is a hang rather than a slow run:
    1e9 per side is 1e18 iterations and was accepted."""
    assert model._MAX_EMITTER_COUNT == 4096
    ok = SourceDefinition('emitter_array', emitter_nx=4096, emitter_ny=1)
    assert ok.emitter_nx == 4096
    for bad in (4097, 1e9, 1e18):
        with pytest.raises(ValueError, match='emitter_nx'):
            SourceDefinition('emitter_array', emitter_nx=bad)
    with pytest.raises(ValueError, match='4096'):
        SourceDefinition('emitter_array', emitter_ny=100000)
    # Non-finite is a count too, and float() accepts it.
    for bad in (float('inf'), float('nan')):
        with pytest.raises(ValueError, match='emitter_nx'):
            SourceDefinition('emitter_array', emitter_nx=bad)


def test_followup_84_mpl_shim_raises_attributeerror_not_importerror():
    """`__dir__` advertises the lazy names, so `hasattr`,
    `inspect.getmembers`, `help()` and a REPL completion all reach the
    PEP 562 hook.  A module `__getattr__` may raise only AttributeError
    there; the raw `ModuleNotFoundError: shiboken6` escaped `hasattr`
    on a box with no Qt bindings instead of making it return False."""
    import inspect as _inspect
    _mpl = UI['_mpl']

    assert 'FigureCanvasQTAgg' in dir(_mpl)
    try:
        present = hasattr(_mpl, 'FigureCanvasQTAgg')   # must not raise
    except Exception as exc:                            # pragma: no cover
        pytest.fail(f'hasattr raised {type(exc).__name__}: {exc}')
    if not present:
        # Headless: the failure must be an AttributeError that still
        # carries the real cause.
        with pytest.raises(AttributeError) as ei:
            _mpl.FigureCanvasQTAgg
        assert ei.value.__cause__ is not None
        assert isinstance(ei.value.__cause__, ImportError)
    # Walking the module must work either way.
    names = [n for n, _ in _inspect.getmembers(_mpl)]
    assert 'style_axes' in names
    # An unknown name is still a plain AttributeError with no cause.
    with pytest.raises(AttributeError):
        _mpl.NotAThing
    # The name that needs no Qt still resolves.
    assert _mpl.Figure is not None


def test_followup_85_inverted_element_range_is_an_error_not_an_empty_run():
    """An inverted Start/End selection used to propagate zero surfaces
    and return a "successful" result -- the source field pushed to the
    focus, i.e. a plausible-looking PSF of nothing."""
    m = SystemModel()
    for k in range(2):
        m.insert_element(1 + k, Element(0, f'L{k+1}', 'Singlet',
                                        distance_mm=20.0 * (k + 1),
                                        surfaces=[SurfaceRow(50.0, 3.0,
                                                             'N-BK7', 12.7),
                                                  SurfaceRow(np.inf, 0.0, '',
                                                             12.7)]))
    m.epd_mm = 5.0
    m._invalidate()

    w, got = _wave_worker({'start_elem': 3, 'end_elem': 1}, lambda: m)
    w.run()
    assert len(got) == 1
    assert 'error' in got[0], got[0]
    assert 'range' in got[0]['error'] and 'start=3' in got[0]['error']

    # A VALID sub-range still runs and still reports itself.
    w, got = _wave_worker({'start_elem': 2, 'end_elem': 2}, lambda: m)
    w.run()
    assert 'error' not in got[0]
    assert got[0]['element_range'].startswith('elements 2..2')


def test_followup_86_interrupted_optimizer_reports_cancelled():
    """`run_optimization` wraps the scipy call in its own
    `except Exception`, and `StopIteration` is an Exception, so the
    sentinel the progress callback raises never reaches the worker's
    `except StopIteration`: an interrupted run reported
    `Merit: ... after N iterations` and read as a completed optimize."""
    O = UI['optimizer_dock']
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.epd_mm = 25.4
    m.opt_variables = [(1, 0, 'radius')]
    m._invalidate()

    w = O.OptimizeWorker(m, max_iter=200,
                         advanced_kwargs={'method': 'Nelder-Mead'})
    w.requestInterruption()
    got, cancels = [], []
    w.finished_result.connect(lambda ok, msg: got.append((ok, msg)))
    w.cancelled.connect(lambda: cancels.append(True))
    w.run()

    assert len(got) == 1 and got[0][0] is False
    assert 'Cancelled' in got[0][1], got[0][1]
    assert cancels == [True]

    # A run that is NOT interrupted still reports its merit normally.
    w2 = O.OptimizeWorker(m, max_iter=5,
                          advanced_kwargs={'method': 'Nelder-Mead'})
    got2 = []
    w2.finished_result.connect(lambda ok, msg: got2.append((ok, msg)))
    w2.run()
    assert len(got2) == 1 and 'Cancelled' not in got2[0][1]


def test_followup_88_non_finite_object_distance_reads_as_infinity():
    """Consumers gate on `object_distance > 0`, which `inf` passes, so
    an infinite conjugate was solved as a finite one.  0.0 is the
    prescription convention for "object at infinity"."""
    for bad in (float('inf'), float('nan'), -5.0, 0.0):
        m = SystemModel()
        m.elements[0].source = SourceDefinition('point_source',
                                                object_distance_mm=bad)
        m._invalidate()
        assert m.object_distance_m() == 0.0, bad
        assert m.to_prescription()['object_distance'] == 0.0, bad

    # A finite form field with no optic placed is still honoured.
    m = SystemModel()
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=250.0)
    m._invalidate()
    assert m.object_distance_m() == pytest.approx(0.250, rel=1e-12)

    # An infinite FIRST-ELEMENT distance falls through to the field.
    # Written straight onto the attribute, deliberately bypassing the
    # mutators -- which now refuse it (see
    # ``test_followup_nonfinite_distance_is_refused_by_the_mutators``).
    # ``recompute_element_frames`` then multiplies inf by the axis
    # vector's zero components and leaves a NaN origin, so the numpy
    # invalid-value warning is expected here and suppressed explicitly
    # rather than hidden.
    m = _point_source_model(100.0, 400.0)
    m.elements[1].distance_mm = float('inf')
    with np.errstate(invalid='ignore'):
        m._invalidate()
        assert m.object_distance_m() == pytest.approx(0.400, rel=1e-12)


def test_followup_89_local_and_world_lists_close_the_same_gaps():
    """`_build_trace_surfaces_internal` closed an element's air gap
    against `elements[ei + 1]` while the world builder used
    `_next_optical_element`.  A surface-less element between two optics
    made them describe different systems: the local list dropped the
    gap the world list kept, so `find_paraxial_focus` and `system_abcd`
    disagreed depending on which list a dock had.
    """
    def _build(with_gap_element):
        m = SystemModel()
        m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                    surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7',
                                                         12.7),
                                              SurfaceRow(np.inf, 0.0, '',
                                                         12.7)]))
        if with_gap_element:
            m.insert_element(2, Element(0, 'EMPTY', 'Singlet',
                                        distance_mm=0.0, surfaces=[]))
        m.insert_element(len(m.elements) - 1,
                         Element(0, 'L2', 'Singlet', distance_mm=40.0,
                                 surfaces=[SurfaceRow(80.0, 3.0, 'N-BK7',
                                                      12.7),
                                           SurfaceRow(np.inf, 0.0, '',
                                                      12.7)]))
        return m

    plain = _build(False)
    with_empty = _build(True)
    wv = plain.wavelength_m

    # Control: an ordinary list is bit-unchanged by the switch.
    assert [s.thickness for s in plain.build_trace_surfaces()] == \
        pytest.approx([0.003, 0.04, 0.003, 0.0], abs=1e-15)

    for tag, m in (('plain', plain), ('surface-less element', with_empty)):
        local = m.build_trace_surfaces()
        world = m.build_trace_surfaces_world()
        assert [s.thickness for s in local] == pytest.approx(
            [s.thickness for s in world], rel=1e-15), tag
        assert find_paraxial_focus(local, wv) == pytest.approx(
            find_paraxial_focus(world, wv), rel=1e-12), tag
        _, efl_l, _, _ = system_abcd(local, wv)
        _, efl_w, _, _ = system_abcd(world, wv)
        assert efl_l == pytest.approx(efl_w, rel=1e-12), tag

    # ... and the two models are the same physical system.
    assert find_paraxial_focus(with_empty.build_trace_surfaces(), wv) == \
        pytest.approx(find_paraxial_focus(plain.build_trace_surfaces(), wv),
                      rel=1e-12)


def _three_element_model():
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(80.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(3, Element(0, 'L3', 'Singlet', distance_mm=25.0,
                                surfaces=[SurfaceRow(120.0, 2.0, 'N-BK7',
                                                     12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    return m


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_followup_nonfinite_distance_is_refused_by_the_mutators(bad):
    """`recompute_element_frames` advances `origin += d * R[:, 2]`, and
    an untilted axis has two zero components, so `inf * 0` is `nan`: a
    single non-finite spacing leaves a NaN `origin` on that element AND
    on every element after it, with nothing to say where it came from.
    `nan` was worse than `inf` -- `max(0, nan)` is `0` in Python, so a
    NaN entry silently moved the element onto the previous one's back
    vertex.

    The three mutators an operator or the optimizer can feed now refuse
    it with a CONVENTIONS §2-prefixed `ValueError` naming the element
    and the value, leave the design untouched, and take no undo step.
    """
    # --- 1. the Distance column, relative mode -------------------------
    m = _three_element_model()
    before = [e.distance_mm for e in m.elements]
    depth = len(m._undo_stack)
    with pytest.raises(ValueError, match='set_display_distance'):
        m.set_display_distance(2, bad)
    assert [e.distance_mm for e in m.elements] == before
    assert len(m._undo_stack) == depth, 'a refused edit left an undo step'
    assert np.all(np.isfinite([np.asarray(e.origin, dtype=float)
                               for e in m.elements]))

    # --- 2. the Distance column, absolute mode -------------------------
    m.set_coordinate_mode('absolute')
    depth = len(m._undo_stack)
    with pytest.raises(ValueError, match='set_display_distance'):
        m.set_display_distance(2, bad)
    assert [e.distance_mm for e in m.elements] == before
    assert len(m._undo_stack) == depth

    # --- 3. the absolute-coordinates editor (Z / X / Y columns) --------
    for col in (3, 6, 7):
        m2 = _three_element_model()
        m2.set_coordinate_mode('absolute')
        depth = len(m2._undo_stack)
        with pytest.raises(ValueError, match='set_element_absolute_field'):
            m2.set_element_absolute_field(2, col, bad)
        assert [e.distance_mm for e in m2.elements] == before, col
        assert len(m2._undo_stack) == depth, col

    # --- 4. the optimizer write-back -----------------------------------
    m3 = _three_element_model()
    m3.opt_variables = [(2, 0, 'distance')]
    m3._invalidate()
    with pytest.raises(ValueError, match='set_variable_values'):
        m3.set_variable_values([bad])
    assert m3.elements[2].distance_mm == pytest.approx(40.0)

    # The message names the element and the value it refused.
    with pytest.raises(ValueError) as ei:
        m3.set_display_distance(2, bad)
    assert "'L2'" in str(ei.value) and repr(bad) in str(ei.value)

    # A non-numeric entry is still refused, by the same helper.
    with pytest.raises(ValueError, match='set_display_distance'):
        m3.set_display_distance(2, 'twelve')


def test_followup_finite_distance_writes_are_bit_identical():
    """Guard on the finiteness check: it must reject and nothing else.

    Every ordinary write goes through unchanged, to the last bit --
    including 0.0, a value already in place, and the `max(0, ...)` clamp
    on a negative entry that predates this guard.
    """
    m = _three_element_model()
    baseline = [float(np.asarray(e.origin, dtype=float)[2])
                for e in m.elements]

    m.set_coordinate_mode('relative')
    for idx, value in ((1, 12.5), (2, 0.0), (3, 25.0), (2, -7.0)):
        m.set_display_distance(idx, value)
        assert m.elements[idx].distance_mm == (value if value > 0 else 0)

    # Absolute mode round trip is still exact (and still bit-exact on an
    # unfolded system, where the axis slope is 1.0 to the last bit).
    m2 = _three_element_model()
    m2.set_coordinate_mode('absolute')
    for idx in (1, 2, 3):
        shown = m2.get_display_distance(idx)
        keep = m2.elements[idx].distance_mm
        m2.set_display_distance(idx, shown)
        assert m2.elements[idx].distance_mm == keep, 'not bit-exact'
    assert [float(np.asarray(e.origin, dtype=float)[2])
            for e in m2.elements] == baseline

    # The absolute editor and the optimizer write-back likewise.
    m3 = _three_element_model()
    m3.set_coordinate_mode('absolute')
    m3.set_element_absolute_field(2, 3, 60.0)
    assert float(np.asarray(m3.elements[2].origin, dtype=float)[2]) == \
        pytest.approx(60.0, rel=1e-12)

    m4 = _three_element_model()
    m4.opt_variables = [(2, 0, 'distance')]
    m4._invalidate()
    m4.set_variable_values([37.5])
    assert m4.elements[2].distance_mm == 37.5
    assert np.all(np.isfinite(np.asarray(m4.elements[3].origin, dtype=float)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
