"""WP-A9 regressions for the designer UI (audit 2026-09-11, findings U1-U7).

PySide6 is NOT installed in this interpreter.  That is NOT a reason to
skip: the auditor's 60-line ``PySide6.QtCore`` stub under
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub`` is
the harness these tests run on, extended here with permissive
``QtWidgets`` / ``QtGui`` shims so the dock modules import.  Everything
asserted below is model-level or pure-function physics/bookkeeping; no
widget is painted.  When a real PySide6 IS present it is used instead and
the same assertions hold.

Oracles, per finding:

* **U1 / U2** -- the UI's OWN internal surface list
  (``build_trace_surfaces()``, the list the 2-D layout and the spot
  diagram are drawn from) versus the exported prescription round-tripped
  through ``raytrace.surfaces_from_prescription``.  The two paths share
  no code past the ``Surface`` dataclass, so agreement is a real
  cross-check, and the layout is by definition what the user sees.
  Pre-fix the exported folded system gave EFL 158.862 mm against the
  layout's 178.774 mm (11.1 % apart) and BFL 156.201 vs 127.802 mm
  (22.2 %); post-fix they agree to 0.0 (bit-identical, same arithmetic
  on the same inputs).
* **U3** -- closed-form direction cosines of a point-source cone:
  ``rho_k = (k/num_rings) * semi_ap / obj_dist`` with azimuths
  ``2*pi*j/rays_per_ring``.  Pre-fix every ray sat on the x = y diagonal
  with ``L = M = rho`` (4 distinct directions out of 25) and the
  marginal ray was ``sqrt(2)`` steep.
* **U4** -- the module globals themselves, read before/after.
* **U6f** -- ``find_paraxial_focus`` on the world list versus on the
  local list; the two describe the same physical system, so any
  disagreement is a bookkeeping bug.  Pre-fix 58.344 mm vs 40.112 mm
  (+45 %).

Error floors: every comparison below is between two float64 evaluations
of the same closed form, so the floor is ~1e-16 relative.  The bars are
set at 1e-12 relative (four decades of headroom) except where exact
equality is the point, and the pre-fix values are decades on the wrong
side of every bar.
"""
import os
import sys
import types
import pathlib

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_REPO = pathlib.Path(__file__).resolve().parents[2]
_STUB = (_REPO / 'docs' / 'audits'
         / 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11' / 'repro' / 'UI' / 'stub')


def _install_qt_stub():
    """Make ``import PySide6`` work headlessly.

    Prefers a real PySide6.  Falls back to the auditor's QtCore stub plus
    permissive QtWidgets / QtGui shims -- enough for the dock modules to
    import and for their module-level functions and unbound methods to be
    exercised.  Never skips: a missing PySide6 is a harness problem, not
    a reason to stop testing the physics glue.
    """
    try:                                    # real PySide6 wins
        import PySide6.QtCore  # noqa: F401
        import PySide6.QtWidgets  # noqa: F401
        import PySide6.QtGui  # noqa: F401
        return 'real'
    except Exception:
        pass

    assert _STUB.is_dir(), (
        f'Qt stub missing at {_STUB}; it is the only harness for the UI '
        f'package on an interpreter without PySide6.')
    if str(_STUB) not in sys.path:
        sys.path.insert(0, str(_STUB))

    import PySide6  # noqa: F401  (the stub package)
    from PySide6 import QtCore

    class _Widget:
        """Permissive widget base: swallows construction and any method."""

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            def _any(*a, **k):
                return None
            return _any

    class _Namespace:
        """Permissive enum / flag namespace (Qt.AlignLeft, ...)."""

        def __getattr__(self, name):
            return 0

    def _permissive(mod):
        """Give ``mod`` a PEP 562 hook that manufactures a dummy class for
        any name a dock module asks for."""
        cache = {}

        def _module_getattr(attr, _cache=cache):
            if attr.startswith('__'):
                raise AttributeError(attr)
            if attr not in _cache:
                _cache[attr] = type(attr, (_Widget,), {})
            return _cache[attr]

        mod.__getattr__ = _module_getattr
        return mod

    for name in ('QtWidgets', 'QtGui', 'QtSvg', 'QtOpenGL'):
        mod = _permissive(types.ModuleType(f'PySide6.{name}'))
        sys.modules[f'PySide6.{name}'] = mod
        setattr(PySide6, name, mod)

    # The QtCore stub is deliberately tiny; make every other name it is
    # asked for resolve to a permissive dummy, and fill in the few the
    # UI relies on behaviourally.
    _permissive(QtCore)
    QtCore.Slot = lambda *a, **k: (lambda f: f)
    QtCore.Qt = _Namespace()
    if not hasattr(QtCore.QThread, 'isInterruptionRequested'):
        QtCore.QThread.isInterruptionRequested = (
            lambda self: bool(getattr(self, '_stub_interrupt', False)))
        QtCore.QThread.requestInterruption = (
            lambda self: setattr(self, '_stub_interrupt', True))
        QtCore.QThread.start = lambda self: None
        QtCore.QThread.wait = lambda self, *a: True
        QtCore.QThread.isRunning = lambda self: False
    return 'stub'


if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np                          # noqa: E402
import pytest                               # noqa: E402

#: Every ``lumenairy.ui`` module these tests exercise.  They are all
#: imported HERE, while the stub is installed, and the stub is then torn
#: back out (see below) so the rest of the pytest session sees exactly
#: the interpreter it had before this module was collected.  Without
#: that teardown, sibling UI test files -- which guard themselves with
#: ``try: import PySide6 / except ImportError: skip`` -- would find the
#: stub, come out of their skip, and fail against a Qt that cannot paint.
_UI_MODULES = (
    'model', '_mpl', '_worker', 'main_window', 'element_table',
    'waveoptics_dock', 'optimizer_dock', 'psf_mtf_dock', 'layout_2d',
    'ao_dock', 'caustic_dock', 'coherence_dock', 'coronagraph_dock',
    'ghost_dock', 'multiconfig_dock', 'phase_retrieval_dock',
    'richards_wolf_dock', 'through_focus_dock', 'tolerance_dock',
    'wavefront_map_dock',
)


def _stub_module_keys():
    return [k for k in sys.modules
            if k == 'PySide6' or k.startswith('PySide6.')
            or k == 'lumenairy.ui' or k.startswith('lumenairy.ui.')]


_PARKED = {}


def _park_stub():
    """Lift the stub (and the ui modules it built) out of
    ``sys.modules``, keeping them alive in ``_PARKED``."""
    for key in _stub_module_keys():
        _PARKED[key] = sys.modules.pop(key)
    if str(_STUB) in sys.path:
        sys.path.remove(str(_STUB))


def _unpark_stub():
    """Put them back, for the duration of one test."""
    sys.modules.update(_PARKED)
    if str(_STUB) not in sys.path:
        sys.path.insert(0, str(_STUB))


def _import_ui_under_stub():
    """Import the UI modules with the stub in place, then park both.

    The module objects stay alive through the references returned here.
    Parking matters because pytest imports every test module during
    COLLECTION: a sibling UI test file that guards itself with
    ``try: import PySide6 / except ImportError: skip`` must make that
    decision against the interpreter it had before this file existed,
    or it comes out of its skip and fails against a Qt that cannot
    paint.  The autouse fixture below un-parks for each of OUR tests.
    """
    flavour = _install_qt_stub()
    import importlib
    mods = {name: importlib.import_module(f'lumenairy.ui.{name}')
            for name in _UI_MODULES}
    if flavour == 'stub':
        _park_stub()
    return flavour, mods


QT_FLAVOUR, UI = _import_ui_under_stub()

from lumenairy.raytrace import (           # noqa: E402
    surfaces_from_prescription, system_abcd, find_paraxial_focus,
)

model = UI['model']
Element = model.Element
SourceDefinition = model.SourceDefinition
SurfaceRow = model.SurfaceRow
SystemModel = model.SystemModel


@pytest.fixture(autouse=True)
def _qt_stub_visible():
    """Make the stub importable for the duration of each test.

    Several code paths re-import lazily (``from .model import
    SystemModel`` inside ``OptimizeWorker._detached_copy``,
    ``MainWindow._ins_source_preset``) and ``inspect.getsource`` looks
    the defining module up in ``sys.modules``; both need the stub live
    while the test runs.  It is parked again afterwards so no other
    test file ever sees it.
    """
    if QT_FLAVOUR == 'real':
        yield
        return
    _unpark_stub()
    try:
        yield
    finally:
        _park_stub()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _folded_model():
    """The audit's ``t2_mirror.py`` system: a concave mirror R = -200 mm
    (semi-diameter 25 mm) followed, at a Zemax-signed -30 mm, by an
    N-BK7 singlet R1 = 80 mm (semi-diameter 6 mm)."""
    m = SystemModel()
    mir = SurfaceRow(radius=-200.0, thickness=0.0, glass='',
                     semi_diameter=25.0, surf_type='Mirror')
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[mir]))
    m.insert_element(2, Element(0, 'Lens 2', 'Singlet', distance_mm=-30.0,
                                surfaces=[
                                    SurfaceRow(80.0, 4.0, 'N-BK7', 6.0),
                                    SurfaceRow(np.inf, 0.0, '', 6.0)]))
    return m


def _singlet_model(distance_mm=10.0):
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=distance_mm,
                                surfaces=[
                                    SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                    SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.epd_mm = 25.4
    m._invalidate()
    return m


def _two_singlet_model(gap_mm=40.0):
    m = SystemModel()
    m.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    m.insert_element(2, Element(0, 'L2', 'Singlet', distance_mm=gap_mm,
                                surfaces=[SurfaceRow(80.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)]))
    return m


# ---------------------------------------------------------------------------
# U1 -- to_prescription() must describe the system the layout shows
# ---------------------------------------------------------------------------

def test_u1_exported_folded_prescription_matches_the_layouts_own_abcd():
    """Oracle: the model's own ``build_trace_surfaces()`` list -- what the
    2-D layout, the spot diagram and ``model.efl_mm`` are computed from.

    Pre-fix (measured on HEAD via repro/UI/t2_mirror.py):
        exported EFL 0.15886162762977662 m, BFL 0.15620131539451035 m
        layout   EFL 0.17877415918110384 m, BFL 0.12780222366203894 m
        -> 11.1 % / 22.2 % apart, because ``is_mirror`` was stripped and
           the Zemax-signed -30 mm post-mirror gap became a literal
           backwards propagation.
    Post-fix the two are the SAME arithmetic on the same inputs, so the
    bar is exact equality; 1e-12 relative is four decades above the
    float64 floor and twelve decades below the pre-fix error.
    """
    m = _folded_model()
    wv = m.wavelength_m

    _, efl_layout, bfl_layout, _ = system_abcd(m.build_trace_surfaces(), wv)
    exported = surfaces_from_prescription(m.to_prescription())
    _, efl_exp, bfl_exp, _ = system_abcd(exported, wv)

    # The pre-fix numbers must be nowhere near.
    assert abs(efl_exp - 0.15886162762977662) > 1e-6, (
        'exported EFL still matches the pre-fix (mirror-stripped) value')

    assert efl_layout == pytest.approx(0.17877415918110384, rel=1e-12)
    assert efl_exp == pytest.approx(efl_layout, rel=1e-12)
    assert bfl_exp == pytest.approx(bfl_layout, rel=1e-12)


def test_u1_is_mirror_and_is_stop_are_emitted_per_surface():
    m = _folded_model()
    pres = m.to_prescription()
    assert [s['is_mirror'] for s in pres['surfaces']] == [True, False, False]
    assert all('is_stop' in s for s in pres['surfaces'])
    # surfaces_from_prescription must pick the flag up.
    surfs = surfaces_from_prescription(pres)
    assert [s.is_mirror for s in surfs] == [True, False, False]


def test_u1_coord_break_air_gap_survives_the_legacy_export():
    """A 45 deg fold (repro/UI/t11.py).  Coord-break Surfaces are folded
    out of the legacy list; their TRANSFER thickness is real axial
    distance and must land in the preceding gap.

    Pre-fix: local list total gap 0.043 m, exported total 0.003 m --
    the 0.040 m mirror-to-lens gap simply vanished.
    """
    m = SystemModel()
    m.insert_element(1, Element(0, 'M1', 'Mirror', distance_mm=50.0,
                                surfaces=[SurfaceRow(np.inf, 0.0, '', 25.0,
                                                     surf_type='Mirror')],
                                tilt_x=45.0))
    m.insert_element(2, Element(0, 'L1', 'Singlet', distance_mm=40.0,
                                surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                          SurfaceRow(np.inf, 0.0, '', 12.7)],
                                tilt_x=45.0))
    local_total = sum(s.thickness for s in m.build_trace_surfaces())
    pres = m.to_prescription()
    assert local_total == pytest.approx(0.043, rel=1e-12)
    assert sum(pres['thicknesses']) == pytest.approx(local_total, rel=1e-12)
    assert pres['thicknesses'][0] == pytest.approx(0.040, rel=1e-12)


def test_u1_singlet_export_stays_byte_equivalent_to_make_singlet():
    """Guard on the U1 change: the unfolded path must not move.

    ``make_singlet`` is written by a different author in a different
    module and shares no code with ``to_prescription``; the audit
    verified byte-equivalence pre-fix and it must survive.
    """
    import lumenairy as la
    m = _singlet_model()
    pres = m.to_prescription()
    lib = la.make_singlet(R1=50e-3, R2=float('inf'), d=3e-3,
                          glass='N-BK7', aperture=25.4e-3)
    assert pres['thicknesses'] == pytest.approx(lib['thicknesses'], rel=1e-15)
    assert pres['aperture_diameter'] == pytest.approx(
        lib['aperture_diameter'], rel=1e-15)
    for a, b in zip(pres['surfaces'], lib['surfaces']):
        for key in ('radius', 'conic', 'glass_before', 'glass_after'):
            assert a[key] == b[key], key
    a_ui, efl_ui, bfl_ui, _ = system_abcd(
        surfaces_from_prescription(pres), m.wavelength_m)
    a_lib, efl_lib, bfl_lib, _ = system_abcd(
        surfaces_from_prescription(lib), m.wavelength_m)
    assert efl_ui == pytest.approx(efl_lib, rel=1e-15)
    assert bfl_ui == pytest.approx(bfl_lib, rel=1e-15)
    # Audit's measured reference values for this fixture.
    assert efl_ui == pytest.approx(0.09928851726861038, rel=1e-12)
    assert bfl_ui == pytest.approx(0.09729328309216069, rel=1e-12)


# ---------------------------------------------------------------------------
# U2 -- per-surface semi-diameters
# ---------------------------------------------------------------------------

def test_u2_every_surface_carries_its_own_semi_diameter():
    """Pre-fix the key was absent entirely, so
    ``surfaces_from_prescription`` fell back to matching the ``elements``
    list -- whose refracting-surface filter skips mirrors while the index
    counts them -- and the trailing surface fell through to
    ``aperture_diameter / 2`` = 12.7 mm instead of its own 6 mm.
    """
    m = _folded_model()
    pres = m.to_prescription()
    got = [s['semi_diameter'] for s in pres['surfaces']]
    assert got == pytest.approx([0.025, 0.006, 0.006], rel=1e-15)

    surfs = surfaces_from_prescription(pres)
    # The two refracting surfaces are now exact (the trailing one was
    # 0.0127 m -- the aperture_diameter/2 fall-through -- pre-fix).
    assert surfs[1].semi_diameter == pytest.approx(0.006, rel=1e-15)
    assert surfs[2].semi_diameter == pytest.approx(0.006, rel=1e-15)
    assert surfs[2].semi_diameter != pytest.approx(0.0127, rel=1e-6)


def test_u2_is_stop_round_trips_through_load_prescription():
    """The model had no place to hold an aperture stop at all, so the
    .zmx STOP flag was dropped on import and could never be exported."""
    m = _singlet_model()
    m.elements[1].surfaces[0].is_stop = True
    pres = m.to_prescription()
    assert pres['surfaces'][0]['is_stop'] is True
    assert pres['stop_index'] == 0
    assert surfaces_from_prescription(pres)[0].is_stop is True

    m2 = SystemModel()
    m2.load_prescription(pres)
    stops = [s.is_stop for e in m2.elements for s in e.surfaces]
    assert stops.count(True) == 1, stops


# ---------------------------------------------------------------------------
# U3 -- the point-source ray bundle
# ---------------------------------------------------------------------------

def _capture_point_source_bundle(model, num_rings, rays_per_ring):
    import lumenairy.raytrace as rt
    captured = {}
    orig = rt._make_bundle

    def spy(x, y, L, M, wv):
        captured['L'] = np.asarray(L, dtype=float)
        captured['M'] = np.asarray(M, dtype=float)
        return orig(x, y, L, M, wv)

    rt._make_bundle = spy
    try:
        model.run_trace(num_rings=num_rings, rays_per_ring=rays_per_ring)
    finally:
        rt._make_bundle = orig
    return captured['L'], captured['M']


def test_u3_point_source_bundle_is_a_real_cone():
    """Closed-form oracle: ring k of ``num_rings`` must reach
    ``rho_k = (k/num_rings) * semi_ap / obj_dist`` at azimuth
    ``2*pi*j/rays_per_ring``.

    Pre-fix measurement (repro/UI/t4_pointsrc.py): 25 rays, **4** unique
    (L, M) pairs, ``L == M`` on every ray, and |rho| over-filled by
    sqrt(2) = 41 % because both cosines carried the full aperture ratio.
    """
    m = _singlet_model(distance_mm=200.0)
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=200.0)
    m.elements[-1].distance_mm = 200.0
    m._invalidate()

    num_rings, rays_per_ring = 3, 8
    L, M = _capture_point_source_bundle(m, num_rings, rays_per_ring)

    assert L.size == num_rings * rays_per_ring + 1
    uniq = {(round(a, 12), round(b, 12)) for a, b in zip(L, M)}
    assert len(uniq) == L.size, f'degenerate bundle: {len(uniq)} of {L.size}'
    assert not np.allclose(L, M), 'rays still collapse onto the x = y diagonal'

    semi_ap = m.epd_m / 2.0
    obj_dist = 200.0e-3
    rho = np.hypot(L, M)
    for k in range(1, num_rings + 1):
        expect = (k / num_rings) * semi_ap / obj_dist
        ring = rho[(k - 1) * rays_per_ring: k * rays_per_ring]
        np.testing.assert_allclose(ring, expect, rtol=1e-12)
    # Marginal ray: exactly semi_ap/obj_dist, not sqrt(2) times it.
    assert rho.max() == pytest.approx(semi_ap / obj_dist, rel=1e-12)
    # Azimuths uniform over the full circle.
    ring0 = np.sort(np.mod(np.arctan2(M[:rays_per_ring],
                                      L[:rays_per_ring]), 2 * np.pi))
    np.testing.assert_allclose(
        ring0, np.arange(rays_per_ring) * 2 * np.pi / rays_per_ring,
        atol=1e-12)
    # The chief ray closes the list.
    assert rho[-1] == pytest.approx(0.0, abs=1e-15)


def test_u3_object_distance_comes_from_the_element_geometry():
    """``SourceDefinition.object_distance_mm`` and the first element's
    ``distance_mm`` are two knobs for one quantity.  The launch must use
    the geometric one, or the pupil fill is wrong by their ratio."""
    m = _singlet_model(distance_mm=100.0)
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=1000.0)
    m._invalidate()
    assert m.object_distance_m() == pytest.approx(0.100, rel=1e-12)
    L, M = _capture_point_source_bundle(m, 2, 4)
    # Geometric obj_dist 100 mm -> rho_max = 12.7 / 100.  Using the
    # stale 1000 mm form field would give a tenth of that.
    assert np.hypot(L, M).max() == pytest.approx(
        (m.epd_m / 2.0) / 0.100, rel=1e-12)
    # A collimated source reports "object at infinity" (0.0).
    m.elements[0].source = SourceDefinition('plane_wave')
    assert m.object_distance_m() == 0.0


# ---------------------------------------------------------------------------
# U4 -- process-global FFT / RAM overrides must not leak out of a run
# ---------------------------------------------------------------------------

def test_u4_fft_backend_and_ram_cap_are_restored_after_a_run():
    """Pre-fix ``_run_impl`` set ``USE_PYFFTW = USE_SCIPY_FFT = False``
    unconditionally, from the worker thread, and never restored them --
    so ONE default Run downgraded every later FFT in the process."""
    from lumenairy.propagators import fft_infra
    from lumenairy.memory import get_max_ram
    _process_overrides = UI['waveoptics_dock']._process_overrides

    before = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT, get_max_ram())

    with _process_overrides('numpy', 4):
        assert fft_infra.USE_PYFFTW is False
        assert fft_infra.USE_SCIPY_FFT is False
        assert get_max_ram() == 4 * 1024 ** 3
    assert (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT,
            get_max_ram()) == before

    with _process_overrides('scipy', None):
        assert fft_infra.USE_SCIPY_FFT is True
        assert fft_infra.USE_PYFFTW is False
    assert (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT,
            get_max_ram()) == before

    # 'default' touches nothing at all -- combo index 0.
    with _process_overrides('default', None):
        assert fft_infra.USE_PYFFTW == before[0]
        assert fft_infra.USE_SCIPY_FFT == before[1]
    assert (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT,
            get_max_ram()) == before


def test_u4_overrides_are_restored_when_the_run_raises():
    from lumenairy.propagators import fft_infra
    from lumenairy.memory import get_max_ram
    _process_overrides = UI['waveoptics_dock']._process_overrides

    before = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT, get_max_ram())
    with pytest.raises(RuntimeError):
        with _process_overrides('numpy', 8):
            raise RuntimeError('boom')
    assert (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT,
            get_max_ram()) == before


# ---------------------------------------------------------------------------
# U5 -- the lens-model router must say what it actually ran
# ---------------------------------------------------------------------------

def test_u5_folded_prescription_is_detected_both_ways():
    _prescription_has_mirror = UI['waveoptics_dock']._prescription_has_mirror

    folded = _folded_model().to_prescription()
    assert _prescription_has_mirror(folded)
    # Detected from the per-surface flag alone (the elements list is what
    # apply_real_lens counts; the surfaces list is what the trace reads).
    assert _prescription_has_mirror({'surfaces': [{'is_mirror': True}]})
    assert _prescription_has_mirror(
        {'elements': [{'element_type': 'mirror'}]})
    assert not _prescription_has_mirror(_singlet_model().to_prescription())
    assert not _prescription_has_mirror({})
    assert not _prescription_has_mirror(None)


def test_u5_router_runs_the_unfolded_equivalent_instead_of_downgrading():
    """``apply_real_lens`` refuses a folded prescription BY DESIGN; the
    router used to catch that and silently run the thin-screen ASM loop
    while still labelling the result with the requested model.

    The dock's "Unfold mirrors" checkbox is the user asking for the
    unfolded equivalent, and ``_filter_wave_optics_surfaces`` already
    builds exactly that.  Feed it to the library instead of catching the
    refusal.
    """
    import warnings
    from lumenairy.elements.lenses import apply_real_lens
    _filter_wave_optics_surfaces = UI['waveoptics_dock']._filter_wave_optics_surfaces
    _prescription_from_surfaces = UI['waveoptics_dock']._prescription_from_surfaces

    m = _folded_model()
    pres = m.to_prescription()
    E = np.ones((32, 32), dtype=complex)

    # The library refuses the folded prescription, as it should.
    with pytest.raises(ValueError, match='mirror'):
        apply_real_lens(E, prescription=pres, wavelength=1.31e-6, dx=4e-6)

    surfs = _filter_wave_optics_surfaces(m.build_trace_surfaces(),
                                         unfold_mirrors=True,
                                         ignore_lateral_cbs=True)
    assert not any(s.is_mirror for s in surfs)
    unfolded = _prescription_from_surfaces(surfs, m.epd_m)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens(E, prescription=unfolded,
                              wavelength=1.31e-6, dx=4e-6)
    assert np.asarray(out).shape == E.shape

    # The helper refuses to pretend a mirror is a refractor.
    with pytest.raises(ValueError, match='still'):
        _prescription_from_surfaces(m.build_trace_surfaces(), m.epd_m)


def test_u5_unfolding_preserves_the_axial_path():
    """Dropping a mirror Surface used to drop its thickness with it, so
    the unfolded path lost the whole mirror-to-next-element gap (30 mm
    on this fixture -- the mirror is the first surface and carries the
    -30 mm Zemax-signed gap)."""
    _filter_wave_optics_surfaces = UI['waveoptics_dock']._filter_wave_optics_surfaces

    m = _two_singlet_model(gap_mm=40.0)
    # Control: with no mirror, unfolding is the identity on thicknesses.
    local = m.build_trace_surfaces()
    kept = _filter_wave_optics_surfaces(local, unfold_mirrors=True,
                                        ignore_lateral_cbs=True)
    assert [s.thickness for s in kept] == pytest.approx(
        [s.thickness for s in local], rel=1e-15)

    # Folded: a mirror between the two singlets.  Its |thickness| must
    # survive on the preceding kept surface.
    mf = SystemModel()
    mf.insert_element(1, Element(0, 'L1', 'Singlet', distance_mm=10.0,
                                 surfaces=[SurfaceRow(50.0, 3.0, 'N-BK7', 12.7),
                                           SurfaceRow(np.inf, 0.0, '', 12.7)]))
    mf.insert_element(2, Element(0, 'M1', 'Mirror', distance_mm=25.0,
                                 surfaces=[SurfaceRow(np.inf, 0.0, '', 25.0,
                                                      surf_type='Mirror')]))
    mf.insert_element(3, Element(0, 'L2', 'Singlet', distance_mm=-40.0,
                                 surfaces=[SurfaceRow(80.0, 3.0, 'N-BK7', 12.7),
                                           SurfaceRow(np.inf, 0.0, '', 12.7)]))
    folded = mf.build_trace_surfaces()
    unfolded = _filter_wave_optics_surfaces(folded, unfold_mirrors=True,
                                            ignore_lateral_cbs=True)
    assert not any(s.is_mirror for s in unfolded)
    # Total unsigned axial path is conserved by the unfolding.
    assert sum(s.thickness for s in unfolded) == pytest.approx(
        sum(abs(s.thickness) for s in folded), rel=1e-12)
    assert sum(s.thickness for s in unfolded) == pytest.approx(
        0.003 + 0.025 + 0.040 + 0.003, rel=1e-12)


# ---------------------------------------------------------------------------
# U6 -- the crash / lambda-mismatch list
# ---------------------------------------------------------------------------

def test_u6a_finished_slices_the_output_grid_not_the_input_grid():
    """``_on_finished`` read ``results['N']`` -- the INPUT grid -- and
    sliced ``I_focus`` with it.  On the three ``*-mft`` methods and on
    any detector run ``I_focus`` is smaller, so the slice was an
    IndexError (the cross-section) or a mis-plot (the imshow).

    Driven through the real method with a minimal recording ``self``.
    """
    WaveOpticsDock = UI['waveoptics_dock'].WaveOpticsDock
    from matplotlib.figure import Figure

    N_in, N_out = 256, 64
    rng = np.random.default_rng(0)
    I_focus = rng.random((N_out, N_out)) + 1e-6

    class _Rec:
        def __init__(self):
            self.text = ''

        def setPlainText(self, t):
            self.text = t

        def setEnabled(self, *a):
            pass

        def setVisible(self, *a):
            pass

        def setText(self, t):
            self.text = t

        def currentText(self):
            return 'ASM'

        def draw(self):
            pass

        def emit(self, *a):
            pass

    fake = types.SimpleNamespace(
        btn_run=_Rec(), btn_stop=_Rec(), progress_bar=_Rec(),
        progress_label=_Rec(), summary=_Rec(), canvas=_Rec(),
        combo_method=_Rec(), combo_backend=_Rec(), run_finished=_Rec(),
        fig=Figure(figsize=(4, 2)), _worker=None)

    results = {
        'I_focus': I_focus, 'dx': 4e-6, 'N': N_in,
        'N_out': N_out, 'dx_out': 1e-6,
        'wavelength': 1.31e-6, 'power_in': 1.0, 'power_focus': 0.5,
        'd4sigma': 1e-5, 'elapsed': 0.1, 'n_planes_saved': 0,
    }
    WaveOpticsDock._on_finished(fake, results)      # must not raise

    ax2 = fake.fig.axes[1]
    xdata = ax2.lines[0].get_xdata()
    # The cross-section spans the OUTPUT grid's central quarter.
    assert len(xdata) == 2 * (N_out // 8)
    # ... and is scaled by dx_out, not by the input dx.
    assert abs(xdata).max() == pytest.approx(
        (N_out // 8) * 1e-6 * 1e6, rel=0.2)
    assert 'in -> ' in fake.summary.text, fake.summary.text


def test_u6b_insert_source_preset_uses_the_setter():
    """``SystemModel.source`` is a read-only property; the six
    Insert > Source presets assigned to it and threw AttributeError out
    of the Qt slot."""
    MainWindow = UI['main_window'].MainWindow

    assert type(SystemModel.source).__name__ == 'property'
    assert SystemModel.source.fset is None, (
        'source gained a setter -- the preset test below no longer '
        'guards anything')

    m = _singlet_model()
    m.set_wavelength(632.8)
    status = types.SimpleNamespace(setText=lambda t: None)
    fake = types.SimpleNamespace(model=m, status_label=status)
    MainWindow._ins_source_preset(fake, 'point_source')     # must not raise

    assert m.source.source_type == 'point_source'
    assert m.source.object_distance_mm == pytest.approx(1000.0)
    assert m.source.wavelength_nm == pytest.approx(632.8)
    for kind in ('plane_wave', 'gaussian', 'gaussian_aperture',
                 'top_hat', 'fiber_mode'):
        MainWindow._ins_source_preset(fake, kind)
        assert m.source.source_type == kind


def test_u6c_source_form_edit_preserves_wavelength_and_polarization():
    """``_apply_source_params`` rebuilt the SourceDefinition from 13 line
    edits, none of which is the wavelength or the polarization, so every
    keystroke reset lambda to 1310 nm and dropped the Jones state."""
    SurfaceDetailPanel = UI['element_table'].SurfaceDetailPanel

    m = _singlet_model()
    m.set_wavelength(632.8)
    m.set_source(SourceDefinition('emitter_array', polarization='rcp'))
    assert m.source.wavelength_nm == pytest.approx(632.8)

    class _Edit:
        def __init__(self, text):
            self._t = text

        def text(self):
            return self._t

    combo = types.SimpleNamespace(currentText=lambda: 'emitter_array')
    fake = types.SimpleNamespace(
        _elem_idx=0, sm=m, src_type_combo=combo,
        src_params={'emitter_pitch_mm': _Edit('0.05'),
                    'emitter_nx': _Edit('12'),
                    'emitter_ny': _Edit('12'),
                    'emitter_waist_mm': _Edit('0.009')},
        _SRC_INT_FIELDS=SurfaceDetailPanel._SRC_INT_FIELDS)
    SurfaceDetailPanel._apply_source_params(fake)

    assert m.source.wavelength_nm == pytest.approx(632.8), (
        'source wavelength reset by a form edit')
    assert m.source.polarization == 'rcp', 'polarization dropped'
    # Counts stay integral so to_source()'s range() works.
    assert isinstance(m.source.emitter_nx, int)
    assert isinstance(m.source.emitter_ny, int)
    src = m.source.to_source(N=32, dx_m=2e-6)
    assert np.asarray(src.E).shape == (32, 32)


def test_u6d_model_wavelength_is_carried_onto_the_source():
    """The worker builds E at the SOURCE's wavelength and propagates at
    the MODEL's.  Nothing kept them equal.

    Oracle: the same point source built at the two wavelengths.  Pre-fix
    max |phase difference| was 0.216 rad (measured, repro/UI/t5_wv.py);
    post-fix it is the float64 floor.
    """
    m = SystemModel()
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=50.0)
    m.set_wavelength(632.8)
    assert m.source.wavelength_nm == pytest.approx(632.8)

    got = m.source.to_source(N=64, dx_m=2e-6)
    want = SourceDefinition('point_source', object_distance_mm=50.0,
                            wavelength_nm=632.8).to_source(N=64, dx_m=2e-6)
    assert got.wavelength == pytest.approx(m.wavelength_m, rel=1e-15)
    dphi = float(np.max(np.abs(np.angle(got.E * np.conj(want.E)))))
    assert dphi < 1e-12, f'phase mismatch {dphi} rad (pre-fix was 0.216)'


def test_u6e_emitter_counts_are_integers():
    """A float count made ``to_source`` raise TypeError, which the
    wave-optics worker swallowed into an EPD-clipped plane wave labelled
    "Emitter array 12x12"."""
    src = SourceDefinition('emitter_array', emitter_nx=float(12),
                           emitter_ny=float(12), emitter_pitch_mm=0.05)
    assert isinstance(src.emitter_nx, int) and src.emitter_nx == 12
    assert np.asarray(src.to_source(N=32, dx_m=2e-6).E).shape == (32, 32)
    with pytest.raises(ValueError, match='emitter_nx'):
        SourceDefinition('emitter_array', emitter_nx=0)
    with pytest.raises(ValueError, match='emitter_ny'):
        SourceDefinition('emitter_array', emitter_ny=3.5)


def test_u6f_world_surface_list_carries_inter_element_air_gaps():
    """``trace_world`` steps between ``world_origin``s and ignores
    ``thickness``, but ``find_paraxial_focus`` / ``system_abcd`` -- which
    six docks and ``run_trace``'s image-plane fallback call on this list
    -- read it.

    Oracle: the LOCAL list, which describes the same physical system.
    Pre-fix (repro/UI/t10_abcd_cb.py) the world list gave
    find_paraxial_focus 0.05834391476238654 m against the local list's
    0.04011208679014488 m -- +45 %, so the detector-at-zero image plane
    landed 18 mm past focus.
    """
    m = _two_singlet_model(gap_mm=40.0)
    wv = m.wavelength_m
    world = m.build_trace_surfaces_world()
    local = m.build_trace_surfaces()

    assert [s.thickness for s in world] == pytest.approx(
        [s.thickness for s in local], rel=1e-15)
    f_world = find_paraxial_focus(world, wv)
    f_local = find_paraxial_focus(local, wv)
    assert f_local == pytest.approx(0.04011208679014488, rel=1e-12)
    assert f_world == pytest.approx(f_local, rel=1e-12)
    assert abs(f_world - 0.05834391476238654) > 1e-6, 'pre-fix value returned'

    _, efl_w, _, _ = system_abcd(world, wv)
    _, efl_l, _, _ = system_abcd(local, wv)
    assert efl_w == pytest.approx(efl_l, rel=1e-12)
    assert efl_l == pytest.approx(0.07297144166448877, rel=1e-12)


def test_u6g_psf_pupil_is_taken_at_the_exit_pupil_not_the_image_plane():
    """(a) ``RayBundle`` has no ``.opl`` -- it is ``.opd``, so the whole
    feature died in the caller's handler.  (b) The ray heights used were
    ``image_rays.x/y`` (positions AT FOCUS, max |r| = 2.50e-04 m for a
    25.4 mm pupil), binned into an EPD-wide grid: 31 of 65536 cells.
    """
    PSFMTFDock = UI['psf_mtf_dock'].PSFMTFDock

    m = _singlet_model(distance_mm=10.0)
    m.elements[-1].distance_mm = 0.0        # paraxial-BFL image plane
    m._invalidate()
    result = m.run_trace(num_rings=8, rays_per_ring=36)

    assert not hasattr(result.image_rays, 'opl')
    assert hasattr(result.image_rays, 'opd')

    img = result.image_rays
    r_img = float(np.max(np.hypot(img.x[img.alive], img.y[img.alive])))
    assert r_img < 1e-3, 'fixture no longer focuses; oracle invalid'

    rays, label = PSFMTFDock._exit_pupil_rays(result)
    assert rays is not None and label != 'Image'
    alive = rays.alive
    r_pup = float(np.max(np.hypot(rays.x[alive], rays.y[alive])))
    # The pupil radius must be the marginal ray height (EPD/2), not the
    # focal spot radius -- two decades apart on this fixture.
    assert r_pup == pytest.approx(m.epd_m / 2.0, rel=0.02)
    assert r_pup > 40 * r_img
    # Exit-vertex referenced: every alive ray is on one plane.
    assert np.allclose(rays.z[alive], 0.0, atol=1e-15)


def test_u6h_optimizer_worker_never_touches_the_live_model():
    """``merit_function`` -> ``set_variable_values`` -> ``_invalidate``
    rewrites every element's ``origin`` / ``R`` and nulls the surface
    cache.  Done on the shared model from the worker thread, that raced
    the GUI's painting on every scipy probe.
    """
    OptimizeWorker = UI['optimizer_dock'].OptimizeWorker

    m = _singlet_model()
    m.opt_variables = [(1, 0, 'radius')]
    m._invalidate()
    live_elem = m.elements[1]
    live_surface = live_elem.surfaces[0]
    r0 = live_surface.radius
    origin0 = np.array(live_elem.origin, dtype=float)

    worker = OptimizeWorker(m, max_iter=3,
                            advanced_kwargs={'method': 'Nelder-Mead'})
    assert worker.model is not m
    assert worker.model.elements[1] is not live_elem
    assert worker.model.elements[1].surfaces[0] is not live_surface

    # Drive the merit the way scipy would; the live model must not move.
    for probe in (r0 * 1.2, r0 * 0.8, r0 * 1.5):
        worker.model.merit_function(np.array([probe]))
    assert live_surface.radius == r0
    np.testing.assert_array_equal(np.asarray(live_elem.origin), origin0)
    assert worker.model.elements[1].surfaces[0].radius == pytest.approx(
        r0 * 1.5)


def test_u6i_workers_honour_request_interruption():
    """14 of 16 workers polled nothing, so ``_shutdown_dock_workers``'s
    ``requestInterruption()`` + ``wait(2000)`` timed out and Qt aborted
    the process mid-run."""
    _w = UI['_worker']
    QThread = _w.QThread
    ThreadCancellableProgress = _w.ThreadCancellableProgress
    interrupt_check = _w.interrupt_check
    AnalysisWorker = _w.AnalysisWorker
    WorkerInterrupted = _w.WorkerInterrupted

    th = QThread()
    prog = ThreadCancellableProgress(th)
    assert prog.should_stop is False
    assert interrupt_check(th) is False
    th.requestInterruption()
    assert prog.should_stop is True, (
        'requestInterruption() must trip the progress object every dock '
        'loop polls')
    assert interrupt_check(th) is True

    # The dock's own Stop button still works independently.
    th2 = QThread()
    prog2 = ThreadCancellableProgress(th2)
    prog2.cancel()
    assert prog2.should_stop is True

    # The base class turns the poll into a clean single emission.
    class _Spin(AnalysisWorker):
        def work(self):
            for _ in range(10):
                self.check_interrupt()
            return {'ok': True}

    w = _Spin()
    got = []
    w.finished_result.connect(got.append)
    w.requestInterruption()
    w.run()
    assert got == [{'error': 'Stopped by user'}]

    class _Boom(AnalysisWorker):
        def work(self):
            raise ValueError('kaboom')

    w2 = _Boom()
    got2 = []
    w2.finished_result.connect(got2.append)
    w2.run()
    assert len(got2) == 1 and 'kaboom' in got2[0]['error']
    assert WorkerInterrupted is not None


def test_u6i_every_qthread_worker_exposes_a_cancellation_path():
    """Structural sweep: each worker must be reachable by
    ``requestInterruption()`` -- either by polling it, by owning a
    ``ThreadCancellableProgress``, or by deriving from
    ``AnalysisWorker``."""
    import inspect
    QThread = UI['_worker'].QThread
    AnalysisWorker = UI['_worker'].AnalysisWorker

    mods = [
        'ao_dock', 'caustic_dock', 'coherence_dock', 'coronagraph_dock',
        'ghost_dock', 'multiconfig_dock', 'optimizer_dock',
        'phase_retrieval_dock', 'psf_mtf_dock', 'richards_wolf_dock',
        'through_focus_dock', 'tolerance_dock', 'wavefront_map_dock',
        'waveoptics_dock',
    ]
    unguarded = []
    n_workers = 0
    for name in mods:
        mod = UI[name]
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not (issubclass(obj, QThread) and obj is not QThread
                    and obj.__module__ == mod.__name__):
                continue
            n_workers += 1
            if issubclass(obj, AnalysisWorker):
                continue
            try:
                src = inspect.getsource(obj)
            except OSError:                       # pragma: no cover
                continue
            if ('isInterruptionRequested' in src
                    or 'interrupt_check' in src
                    or 'ThreadCancellableProgress' in src):
                continue
            unguarded.append(f'{name}.{obj.__name__}')
    assert n_workers >= 14, f'only found {n_workers} workers'
    assert not unguarded, f'workers with no cancellation path: {unguarded}'


def test_u6_no_worker_shadows_the_builtin_finished_signal():
    """Three QThread subclasses redefined ``finished``, so
    ``worker.finished.connect(worker.deleteLater)`` bound to the custom
    signal -- wrong payload, and never emitted when ``run()`` raised."""
    import inspect
    QThread = UI['_worker'].QThread

    shadowing = []
    for name in ('optimizer_dock', 'tolerance_dock', 'waveoptics_dock',
                 'coronagraph_dock', 'ghost_dock', 'caustic_dock',
                 'ao_dock', 'coherence_dock', 'multiconfig_dock',
                 'phase_retrieval_dock', 'psf_mtf_dock',
                 'richards_wolf_dock', 'through_focus_dock',
                 'wavefront_map_dock'):
        mod = UI[name]
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not (issubclass(obj, QThread) and obj is not QThread
                    and obj.__module__ == mod.__name__):
                continue
            if 'finished' in vars(obj):
                shadowing.append(f'{name}.{obj.__name__}')
    assert not shadowing, (
        f'these workers shadow QThread.finished: {shadowing}')


# ---------------------------------------------------------------------------
# U7 -- the P2 / P3 list
# ---------------------------------------------------------------------------

def test_u7_explicit_image_distance_beats_the_detector():
    """``spot_field_dock`` computes the paraxial BFL and passes it, then
    draws a focal-plane Airy overlay -- on a spot diagram that was
    silently rendered at the detector plane instead."""
    m = _singlet_model()
    m.elements[-1].distance_mm = 100.0          # detector well past focus
    m._invalidate()
    bfl = find_paraxial_focus(m.build_trace_surfaces(), m.wavelength_m)
    assert bfl == pytest.approx(0.09729328309216069, rel=1e-9)

    world = m.build_trace_surfaces_world()
    last = world[-1]
    asked = m.build_run_trace_world_surfaces(image_distance=bfl)
    z_img = float(asked[-1].world_origin[2])
    z_expected = float(last.world_origin[2] + bfl * last.world_R[2, 2])
    assert z_img == pytest.approx(z_expected, rel=1e-12)
    # ... and is NOT the detector plane (100 mm vs 97.29 mm).
    det_z = float(np.asarray(m.elements[-1].origin)[2]) * 1e-3
    assert abs(z_img - det_z) > 2e-3

    # Default (None) still prefers the detector -- unchanged behaviour.
    default = m.build_run_trace_world_surfaces()
    assert float(default[-1].world_origin[2]) == pytest.approx(det_z,
                                                               rel=1e-12)


def test_u7_stale_optimization_variables_cannot_misassign():
    """``get_variable_values`` skipped stale entries while
    ``set_variable_values`` indexed the unfiltered list, so deleting an
    element that owned a variable wrote values into the wrong
    parameter -- or raised IndexError, swallowed into a generic
    failure."""
    m = _two_singlet_model()
    m.opt_variables = [(1, 0, 'radius'), (99, 0, 'radius'),
                       (2, 0, 'radius')]
    live = m.live_opt_variables()
    assert live == [(1, 0, 'radius'), (2, 0, 'radius')]
    x = m.get_variable_values()
    assert len(x) == len(live) == 2
    m.set_variable_values([11.0, 22.0])
    assert m.elements[1].surfaces[0].radius == 11.0
    assert m.elements[2].surfaces[0].radius == 22.0
    # A wrongly sized vector is now a named error, not a silent
    # off-by-one write.
    with pytest.raises(ValueError, match='set_variable_values'):
        m.set_variable_values([1.0, 2.0, 3.0])

    # Deleting an element re-bases the indices instead of stranding them.
    m2 = _two_singlet_model()
    m2.opt_variables = [(1, 0, 'radius'), (2, 0, 'radius')]
    m2.delete_element(1)
    assert m2.opt_variables == [(1, 0, 'radius')]
    assert len(m2.get_variable_values()) == 1


def test_u7_min_thickness_merit_has_no_constant_offset():
    """Every element's trailing surface has thickness 0 by the model's own
    convention (its air gap lives on ``Element.distance_mm``), so
    penalising it added a fixed ``(1 - 0)**2 = 1`` per element and the
    merit could never reach 0."""
    m = _two_singlet_model()
    m.geo_merit_type = 'min_thickness'
    for e in m.elements[1:-1]:
        e.surfaces[0].thickness = 5.0          # well over the 1 mm floor
    m._invalidate()
    m.opt_variables = [(1, 0, 'radius')]
    x = m.get_variable_values()
    assert m.merit_function(x) == pytest.approx(0.0, abs=1e-15)

    # A genuine violation is still penalised, by exactly (1 - t)**2.
    m.elements[1].surfaces[0].thickness = 0.25
    assert m.merit_function(x) == pytest.approx((1.0 - 0.25) ** 2, rel=1e-12)


def test_u7_asm_equiv_reproduces_its_own_calibration_points():
    """A pure cost-model coefficient for the forecast's Time estimate.
    Its docstring's two calibration points are the oracle; the shipped
    formula returned 1.4 and 2.6 against the stated 1.1 and 2.2."""
    _apply_real_lens_asm_equiv = UI['waveoptics_dock']._apply_real_lens_asm_equiv

    assert _apply_real_lens_asm_equiv(2) == pytest.approx(1.1, rel=1e-12)
    assert _apply_real_lens_asm_equiv(3) == pytest.approx(2.2, rel=1e-12)
    assert _apply_real_lens_asm_equiv(1) > 0


def test_u7_file_new_keeps_display_preferences():
    """``File > New`` re-ran ``SystemModel.__init__()`` on the live
    QObject, wiping prefs / lens_options / auto_retrace_mode /
    unit_preference along with the design."""
    m = _two_singlet_model()
    m.prefs['ray_color'] = '#ff0000'
    m.prefs['theme'] = 'solarized'
    m.lens_options['apply_real_lens'] = {'bandlimit': False}
    m.auto_retrace_mode = 'manual'
    m.set_wavelength(632.8)
    m.opt_variables = [(1, 0, 'radius')]

    m.reset_design()

    assert m.prefs['ray_color'] == '#ff0000'
    assert m.prefs['theme'] == 'solarized'
    assert m.lens_options['apply_real_lens'] == {'bandlimit': False}
    assert m.auto_retrace_mode == 'manual'
    # ... and the DESIGN really is gone.
    assert [e.elem_type for e in m.elements] == ['Source', 'Detector']
    assert m.opt_variables == []
    assert m.wavelength_nm == pytest.approx(1310.0)


def test_u7_absolute_z_column_round_trips():
    """``get_display_distance`` summed ``distance_mm`` alone while
    ``set_display_distance`` subtracted the previous element's internal
    thickness, so the two were not inverses: typing the displayed value
    back moved the element."""
    m = _two_singlet_model(gap_mm=40.0)
    m.set_coordinate_mode('absolute')
    for idx in (1, 2):
        shown = m.get_display_distance(idx)
        # The absolute column must agree with the frame the layouts use.
        assert shown == pytest.approx(
            float(np.asarray(m.elements[idx].origin)[2]), rel=1e-12)
        before = m.elements[idx].distance_mm
        m.set_display_distance(idx, shown)      # write back what we read
        assert m.elements[idx].distance_mm == pytest.approx(before,
                                                            abs=1e-9)


def test_u7_bulk_edit_groups_one_undo_step():
    """``_suppress_history`` was set False in ``__init__``, read in
    ``_checkpoint`` and never set True anywhere -- a dead flag."""
    m = _two_singlet_model()
    depth0 = len(m._undo_stack)
    with m.bulk_edit():
        m.set_wavelength(600.0)
        m.set_epd(30.0)
        m.set_wavelength(700.0)
    assert len(m._undo_stack) == depth0 + 1
    m.undo()
    assert m.wavelength_nm == pytest.approx(1310.0)


def test_u7_session_round_trip_keeps_polarization_and_stop():
    """``enc_source`` listed 12 of the 16 constructor kwargs: the
    polarization, the top-hat diameter and the two fiber-mode fields
    were silently reset to defaults by a save/restore."""
    m = _singlet_model()
    m.set_source(SourceDefinition('fiber_mode', fiber_mfd_um=9.2,
                                  fiber_NA=0.12, polarization='linear_45'))
    m.elements[1].surfaces[0].is_stop = True

    state = m._capture_state()
    restored = SystemModel._state_from_jsonable(
        SystemModel._state_to_jsonable(state))
    src = restored['elements'][0].source
    assert src.polarization == 'linear_45'
    assert src.fiber_mfd_um == pytest.approx(9.2)
    assert src.fiber_NA == pytest.approx(0.12)
    assert restored['elements'][1].surfaces[0].is_stop is True


def test_u7_analysis_image_plane_wfe_block_is_reachable():
    """``analysis.py``'s Image-plane WFE panel was gated on
    ``presc.get('object_distance', 0) > 0`` and ``to_prescription()``
    never emitted that key, so the whole block was dead."""
    m = _singlet_model(distance_mm=100.0)
    m.elements[0].source = SourceDefinition('point_source',
                                            object_distance_mm=100.0)
    m._invalidate()
    pres = m.to_prescription()
    assert pres.get('object_distance', 0) > 0
    assert pres['object_distance'] == pytest.approx(0.100, rel=1e-12)

    # Collimated designs keep the infinite conjugate (block stays off).
    m.elements[0].source = SourceDefinition('plane_wave')
    m._invalidate()
    assert m.to_prescription().get('object_distance', 0) == 0.0


def test_u7_ui_package_is_not_on_the_import_lumenairy_path():
    """The audit verified ``import lumenairy`` does not pull
    ``lumenairy.ui``; keep it that way."""
    import subprocess
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys; import lumenairy; '
         'print("ui" in sys.modules.get("lumenairy", lumenairy).__dict__, '
         '"lumenairy.ui" in sys.modules)'],
        cwd=str(_REPO), capture_output=True, text=True,
        stdin=subprocess.DEVNULL,
        env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False False', out.stdout


def test_u7_matplotlib_is_not_imported_by_the_dock_modules():
    """30 dock modules pulled matplotlib at module scope; a dock the user
    never opens should not pay for it (17.5 s cumulative cold on this
    box, sub-second warm).

    Runs in a FRESH interpreter -- in this one the earlier tests have
    already constructed figures -- and bootstraps the same Qt stub by
    importing this module.
    """
    import subprocess
    probe = (
        'import sys\n'
        f'sys.path.insert(0, r"{_REPO / "tests" / "unit"}")\n'
        f'sys.path.insert(0, r"{_REPO}")\n'
        'import test_audit2609_a9_ui as _h\n'
        '_h._unpark_stub()   # the harness parks it again after import\n'
        'import importlib\n'
        'for _k in list(sys.modules):\n'
        '    if _k.startswith("lumenairy.ui"):\n'
        '        del sys.modules[_k]\n'
        'before = "matplotlib.figure" in sys.modules\n'
        'for m in ("caustic_dock","ghost_dock","rayfan_dock",'
        '"spot_field_dock","tolerance_dock","jones_pupil_dock",'
        '"waveoptics_dock","coherence_dock","distortion_dock",'
        '"footprint_dock","glass_map_dock","lg_aberration_dock",'
        '"richards_wolf_dock","shack_hartmann_dock","thin_grating_dock"):\n'
        '    importlib.import_module("lumenairy.ui." + m)\n'
        'after = "matplotlib.figure" in sys.modules\n'
        'print(before, after)\n')
    out = subprocess.run(
        [sys.executable, '-c', probe], cwd=str(_REPO),
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    if out.returncode != 0:
        pytest.fail(f'dock import probe failed:\n{out.stderr}')
    before, after = out.stdout.strip().split()[-2:]
    assert before == 'False', 'harness itself pulled matplotlib'
    assert after == 'False', (
        f'matplotlib still imported at dock-module scope: {out.stdout}')


def test_u7_lazy_mpl_shim_resolves_and_styles():
    _mpl = UI['_mpl']

    fig = _mpl.Figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    _mpl.style_axes(ax, grid=True)
    assert ax.get_facecolor()[:3] == pytest.approx(
        tuple(int(_mpl.BG[i:i + 2], 16) / 255 for i in (1, 3, 5)), abs=1e-6)
    with pytest.raises(AttributeError):
        _mpl.NotAThing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
