"""WP-A21: two diagnostics that named something that does not exist.

Both are the same defect in different clothing -- a string a user is told to
act on, naming a thing they cannot find:

* **WP-A18 section 8.2.**  The 2-D RCWA Wood-anomaly diagnostic passed
  ``fn_name="RCWA2DPrepared.solve"``.  There is no ``RCWA2DPrepared``; the
  class is ``PreparedRCWA2D`` (``rcwa/twod.py``, built by ``prepare_rcwa_2d``).
  A user who greps the warning text for the class finds nothing, and the
  Migration-Guide had to document both spellings.
* **WP-A3 section 5.8, re-raised by WP-A18 section 8.3.**  The designer's
  ``fast_analytic_phase`` checkbox promised "~25 % speedup with <10 nm OPL
  error on typical refractive prescriptions".  The error is not a flat 10 nm:
  it is the OMITTED IN-GLASS ASM LEG, so it scales with centre thickness --
  ~7 nm rms PER MM OF GLASS, which is what ``docs/subsystems/real_lens.md``
  section 4 and ``_lens_traced._geometric_lens_phase``'s docstring now say.
  (The speedup half was re-measured here too; see the test.)

PySide6 is NOT installed in this interpreter.  That is not a reason to skip:
the auditor's Qt stub under ``docs/audits/.../repro/UI/stub`` is the harness,
installed and then PARKED exactly as ``test_audit2609_a9_ui.py`` does it, so
that sibling UI files -- which guard themselves with ``try: import PySide6 /
except ImportError: skip`` -- make that decision against the interpreter they
had before this file was collected.
"""
import os
import pathlib
import sys
import types
import warnings

import numpy as np
import pytest

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_REPO = pathlib.Path(__file__).resolve().parents[2]
_STUB = (_REPO / 'docs' / 'audits'
         / 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11' / 'repro' / 'UI' / 'stub')


# ===========================================================================
# WP-A18 section 8.2 -- the Wood-anomaly diagnostic names a real class
# ===========================================================================

def _wood_2d_prepared():
    """A prepared 2-D solve sitting EXACTLY on a Wood anomaly.

    ``period_x = period_y = lambda = 1 um`` with air on both sides puts the
    ``(+/-1, 0)`` and ``(0, +/-1)`` orders exactly at cut-off (``kt^2 = 1 =
    eps``) -- the canonical Moharam mount, in 2-D.  ``n_orders 3x3`` on a 16x16
    cell keeps it at a few hundred ms.
    """
    from lumenairy.elements.rcwa.twod import prepare_rcwa_2d
    s = 16
    cell = np.ones((s, s), dtype=complex)
    cell[: s // 2, : s // 2] = 2.04 ** 2
    prep = prepare_rcwa_2d(1.0e-6, 1.0e-6, cell, 1.0, 1.0, 1.0e-6,
                           n_orders_x=3, n_orders_y=3, formulation='laurent')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prep.solve(1.0e-6)
    return caught


def test_the_wood_anomaly_diagnostic_names_a_class_that_exists():
    """The class named in the warning must RESOLVE on the module it comes from.

    This is deliberately not a string comparison against the literal
    ``'PreparedRCWA2D'``: that would pass on any future rename that updated the
    literal and forgot the class, which is exactly the failure being fixed.
    The assertion is that the ``Class.method`` the diagnostic prints can be
    looked up -- ``getattr(twod, 'PreparedRCWA2D')`` and then ``.solve`` on it.
    Pre-fix the warning said ``RCWA2DPrepared.solve`` and the first ``getattr``
    raised ``AttributeError``.
    """
    from lumenairy.elements.rcwa import twod as _twod
    from lumenairy.elements.rcwa._core import WoodNudgeWarning

    caught = _wood_2d_prepared()
    wood = [w for w in caught if issubclass(w.category, WoodNudgeWarning)]
    assert len(wood) == 1, [str(w.message)[:70] for w in caught]

    head = str(wood[0].message).split(':', 1)[0]
    assert '.' in head, (
        f'the diagnostic no longer leads with a Class.method name: {head!r}')
    cls_name, meth = head.split('.', 1)
    cls = getattr(_twod, cls_name, None)
    assert cls is not None, (
        f'the Wood-anomaly diagnostic names {cls_name!r}, which does not '
        f'exist on lumenairy.elements.rcwa.twod.  A user who greps the '
        f'warning text for the class finds nothing.  (Pre-fix this was '
        f'"RCWA2DPrepared"; the class is "PreparedRCWA2D".)')
    assert callable(getattr(cls, meth, None)), (
        f'{cls_name} exists but has no callable {meth!r}')
    # ... and it is the class the builder actually returns.
    assert cls.__name__ in _twod.__all__, (
        f'{cls.__name__} is not in twod.__all__, so the name the warning '
        f'prints is not a public one')


def test_no_dead_class_name_survives_in_the_rcwa_2d_diagnostics():
    """Counter-pin: every ``fn_name`` the 2-D module hands the Wood helpers
    names something importable.

    ``rcwa/twod.py`` passes four different ``fn_name`` strings into
    ``_grazing_safe_wavelength_pair`` / ``_WoodAnomaly``.  Three are module
    functions and one is a bound method; all four must resolve.  A static walk
    is the right shape here -- exercising all four would mean four Wood-mount
    solves for a property that is a name lookup.
    """
    import ast
    import inspect

    from lumenairy.elements.rcwa import twod as _twod

    src = inspect.getsource(_twod)
    names = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        target = getattr(f, 'id', None) or getattr(f, 'attr', None)
        if target not in ('_grazing_safe_wavelength_pair', '_WoodAnomaly'):
            continue
        for kw in node.keywords:
            if kw.arg == 'fn_name' and isinstance(kw.value, ast.Constant):
                names.add(kw.value.value)
        for arg in node.args[:1]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)

    assert len(names) >= 4, (
        f'expected at least four fn_name literals in rcwa/twod.py, found '
        f'{sorted(names)} -- the walk has stopped seeing the call sites')
    bad = []
    for n in sorted(names):
        head = n.split('.')
        obj = getattr(_twod, head[0], None)
        for part in head[1:]:
            obj = getattr(obj, part, None) if obj is not None else None
        if obj is None:
            bad.append(n)
    assert not bad, (
        f'these fn_name strings in rcwa/twod.py name nothing importable from '
        f'the module: {bad}.  The diagnostic is the only handle a user has on '
        f'where a Wood nudge came from.')


# ===========================================================================
# WP-A3 section 5.8 / WP-A18 section 8.3 -- the fast_analytic_phase tooltip
# ===========================================================================

def _install_qt_stub():
    """Make ``import PySide6`` work headlessly.  Prefers a real PySide6."""
    try:
        import PySide6.QtCore     # noqa: F401
        import PySide6.QtGui      # noqa: F401
        import PySide6.QtWidgets  # noqa: F401
        return 'real'
    except Exception:             # noqa: BLE001  (any Qt import failure)
        pass

    assert _STUB.is_dir(), (
        f'Qt stub missing at {_STUB}; it is the only harness for the UI '
        f'package on an interpreter without PySide6.')
    if str(_STUB) not in sys.path:
        sys.path.insert(0, str(_STUB))

    import PySide6                # noqa: F401  (the stub package)
    from PySide6 import QtCore

    class _Widget:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            def _any(*a, **k):
                return None
            return _any

    class _Namespace:
        def __getattr__(self, name):
            return 0

    def _permissive(mod):
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
    _permissive(QtCore)
    QtCore.Slot = lambda *a, **k: (lambda f: f)
    QtCore.Qt = _Namespace()
    return 'stub'


_PARKED = {}


def _stub_module_keys():
    return [k for k in sys.modules
            if k == 'PySide6' or k.startswith('PySide6.')
            or k == 'lumenairy.ui' or k.startswith('lumenairy.ui.')]


def _park_stub():
    for key in _stub_module_keys():
        _PARKED[key] = sys.modules.pop(key)
    if str(_STUB) in sys.path:
        sys.path.remove(str(_STUB))


def _unpark_stub():
    sys.modules.update(_PARKED)
    if str(_STUB) not in sys.path:
        sys.path.insert(0, str(_STUB))


def _import_dialog_under_stub():
    """Import the dialog module with the stub in place, then park both.

    Parking matters because pytest imports every test module during
    COLLECTION: a sibling UI test file that guards itself with
    ``try: import PySide6 / except ImportError: skip`` must make that decision
    against the interpreter it had before this file existed.
    """
    flavour = _install_qt_stub()
    import importlib
    mod = importlib.import_module('lumenairy.ui.lens_options_dialog')
    if flavour == 'stub':
        _park_stub()
    return flavour, mod


QT_FLAVOUR, DIALOG = _import_dialog_under_stub()


@pytest.fixture(autouse=True)
def _qt_stub_visible():
    if QT_FLAVOUR == 'real':
        yield
        return
    _unpark_stub()
    try:
        yield
    finally:
        _park_stub()


def _tooltip(func, kwarg):
    for name, _kind, _default, extra in DIALOG.LENS_KWARG_REGISTRY[func]:
        if name == kwarg:
            return extra['tooltip']
    raise AssertionError(f'{func}/{kwarg} is not in LENS_KWARG_REGISTRY')


def test_the_fast_analytic_phase_tooltip_states_the_per_mm_error():
    """The tooltip must carry the PER-MM form, which is the measured one.

    ORACLE: ``lumenairy.elements._lens_traced._geometric_lens_phase``'s own
    docstring, which carries WP-A3's measurement -- 0.007 nm rms at 1 um of
    centre thickness, 0.7 nm at 100 um, 14.1 nm rms / 41.6 nm PV at 2 mm on an
    N-BK7 100/-100 biconvex at f/12.1, i.e. LINEAR IN CENTRE THICKNESS and
    ``~7 nm rms per mm of glass``.  The same sentence is in
    ``docs/subsystems/real_lens.md`` section 4.  Asserting that the tooltip and
    the docstring agree on the phrase is what keeps the two from drifting
    again: WP-A18 recorded them in disagreement.

    The pre-fix text is asserted ABSENT, not just the new text present -- a
    tooltip that said both would be worse than one that said neither.
    """
    tip = _tooltip('apply_real_lens_traced', 'fast_analytic_phase')
    low = tip.lower()

    assert 'per mm of glass' in low, (
        f'the tooltip does not state the per-mm form of the error: {tip!r}')
    assert '7 nm' in tip, (
        f'the tooltip does not carry the measured ~7 nm rms figure: {tip!r}')
    # The retired claims.
    assert '<10 nm' not in tip.replace(' ', ''), (
        f'the tooltip still promises a FLAT <10 nm OPL error, which is wrong '
        f'above ~1.5 mm of glass whatever the f-number: {tip!r}')
    assert '25%' not in tip.replace(' ', ''), (
        f'the tooltip still promises a ~25 % speedup.  MEASURED 2026-09-12 on '
        f'a 2 mm N-BK7 singlet, medians of 9 interleaved pairs: 0.87x-1.00x '
        f'with the parallel amp pass ON (it already overlaps the reference '
        f'leg, so there is nothing to save) and 1.06x-1.17x with it off.  '
        f'Got: {tip!r}')

    # ... and the source of truth says the same thing.
    from lumenairy.elements import _lens_traced
    doc = _lens_traced._geometric_lens_phase.__doc__ or ''
    assert 'per mm of glass' in doc.lower(), (
        'the tooltip now states the per-mm error but _geometric_lens_phase no '
        'longer does -- the two must not drift apart again')


def test_the_tooltip_registry_is_the_dialog_single_source_of_truth():
    """Guard the oracle above: the tooltip has to be reachable where the
    dialog reads it, not just present somewhere in the file.

    ``LENS_KWARG_REGISTRY`` is the module docstring's stated single source of
    truth for what the dialog exposes -- a tooltip corrected anywhere else
    would not reach a user.
    """
    reg = DIALOG.LENS_KWARG_REGISTRY
    assert 'apply_real_lens_traced' in reg
    kwargs = {name for name, _k, _d, _e in reg['apply_real_lens_traced']}
    assert 'fast_analytic_phase' in kwargs
    for func, rows in reg.items():
        for name, kind, _default, extra in rows:
            assert kind in ('bool', 'int', 'float', 'enum', 'str'), (
                f'{func}/{name} declares widget kind {kind!r}')
            assert isinstance(extra.get('tooltip', ''), str)
