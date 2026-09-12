"""WP-A15b -- ``lumenairy/backend/_optional.py`` and ``lumenairy/_knobs.py``.

Covers two findings of ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11``
section 14 (TESTS-ARCH):

* **P2-9** -- ``_ensure_cupy_loaded`` / ``_is_cupy_array`` / ``_load_numba``
  each existed in FIVE hand-copied versions, so the accelerator-absent path
  had five implementations and no single place to test it.  The shared
  ``backend/_optional.py`` is that place, and the parity tests below pin that
  the non-lens consumers still behave exactly as their copies did.
* **P2-5** -- 53 ``set_*`` verbs, 0 context-manager forms, 0 resets, in a
  suite run serially: a leaked process global silently changes a later test
  and does not reproduce under ``-k``.  ``lumenairy.override(...)`` and the
  ``_knobs`` registry close that class.

Oracles
-------
For ``_optional`` the oracle is the pre-refactor source itself: the five
copies' semantics are reproduced here as a small reference implementation
(``_reference_is_cupy_array``) and cross-checked against both the shared
helper and each consumer's surviving wrapper, so "bit-identical" is asserted
rather than asserted-by-inspection.

For ``_knobs`` the oracles are algebraic: ``restore(snapshot())`` is the
identity on every knob, ``override`` is the identity on exit for every exit
path (return / exception / nested), and the restore ORDER is observable
through a probe knob that records the sequence of setter calls.

CuPy is NOT installed on the calibration box (verified by the first test), so
the CuPy arm is exercised at the only point that matters without it: every
predicate must answer False WITHOUT importing anything.  The numba arm runs
for real -- numba IS installed here.
"""
from __future__ import annotations

import concurrent.futures
import importlib
import importlib.util
import sys

import numpy as np
import pytest

import lumenairy as la
from lumenairy import _knobs
from lumenairy.backend import _optional


# ===========================================================================
# backend/_optional.py -- the shared probes
# ===========================================================================

def _reference_is_cupy_array(x, available, cp_module):
    """The five copies' logic, transcribed from the pre-refactor source.

    ``_lens_real.py:44``, ``_lens_thin.py:41``, ``_lens_traced.py:48``,
    ``lenses.py:121`` and ``fft_infra.py:56`` were byte-equivalent to::

        if not CUPY_AVAILABLE:            return False
        if cp is None and not _ensure_cupy_loaded():  return False
        return isinstance(x, cp.ndarray)
    """
    if not available:
        return False
    if cp_module is None:
        return False
    return isinstance(x, cp_module.ndarray)


class TestOptionalCupy:

    def test_availability_flag_matches_find_spec_and_imports_nothing(self):
        """The probe is ``find_spec``, which does not import the package."""
        expected = importlib.util.find_spec('cupy') is not None
        assert _optional.CUPY_AVAILABLE is expected
        if not expected:
            assert 'cupy' not in sys.modules

    def test_absence_is_a_value_not_an_exception(self):
        """CONVENTIONS section 10: the accelerator-absent path is the one
        every NumPy user takes, so it must not raise and must not import.

        No ``pytest.skip`` on the resource precondition (TESTING_STANDARDS
        S3): BOTH arms carry an assertion, so the test discriminates on any
        box.  On a CUDA box it pins that the loader really returns the
        module; here (no cupy) it pins that nothing is imported.
        """
        never_raises = (np.zeros(3), None, object(), 1.0, 'x')
        for obj in never_raises:
            assert _optional.is_cupy_array(obj) is False or \
                _optional.CUPY_AVAILABLE
        if _optional.CUPY_AVAILABLE:             # pragma: no cover - no CUDA
            assert _optional.ensure_cupy() is not None
            assert _optional.cupy_module() is _optional.ensure_cupy()
            assert _optional.is_cupy_array(np.zeros(3)) is False
        else:
            assert _optional.ensure_cupy() is None
            assert _optional.cupy_module() is None
            for obj in never_raises:
                assert _optional.is_cupy_array(obj) is False
            assert 'cupy' not in sys.modules, (
                'is_cupy_array imported cupy on a box where find_spec said '
                'it is absent')

    def test_matches_the_reference_implementation_on_numpy_input(self):
        """Parity with the five copies' transcribed logic."""
        for obj in (np.zeros(3), np.zeros((2, 2), dtype=np.complex128),
                    None, 1.0, 'x', object()):
            assert _optional.is_cupy_array(obj) == _reference_is_cupy_array(
                obj, _optional.CUPY_AVAILABLE, _optional.cupy_module())

    def test_numpy_2x_device_attribute_is_not_mistaken_for_cupy(self):
        """The defect the copies' comments all cite: NumPy 2.x exposes
        ``ndarray.device`` (Array API), so a ``hasattr(x, 'device')``
        duck-type test routes every NumPy array into the CuPy branch.

        FAIL-BEFORE: ``hasattr(np.zeros(3), 'device')`` is True on the
        installed NumPy, so the historical predicate would return True here
        where the shipped one returns False.
        """
        a = np.zeros(3)
        assert hasattr(a, 'device'), (
            'this pin assumes NumPy >= 2.0 array-API ndarray.device; on an '
            'older NumPy the defect it guards cannot occur')
        assert _optional.is_cupy_array(a) is False


@pytest.mark.parametrize('modname,attr', [
    ('lumenairy.propagators.fft_infra', '_is_cupy_array'),
    ('lumenairy.propagators.fft_infra', '_ensure_cupy_loaded'),
    ('lumenairy.sources.core', '_ensure_cupy_loaded'),
])
def test_switched_sites_keep_their_public_spelling(modname, attr):
    """The three non-lens sites now delegate, but their module-level names
    are load-bearing: ``fft_infra.__all__`` lists ``_ensure_cupy_loaded`` /
    ``_is_cupy_array`` and ``propagation.py`` re-exports both."""
    mod = importlib.import_module(modname)
    assert callable(getattr(mod, attr))


def test_fft_infra_keeps_its_cp_alias_contract():
    """``fft_infra`` reads the MODULE-LEVEL ``cp`` inside its CuPy branches
    (``cp.fft.fft2(x)``), so ``_is_cupy_array(x) is True`` must imply ``cp``
    is bound.  The delegation must not break that coupling."""
    from lumenairy.propagators import fft_infra as fi
    assert hasattr(fi, 'cp')
    if not fi.CUPY_AVAILABLE:
        assert fi.cp is None
        assert fi._ensure_cupy_loaded() is False
        assert fi._is_cupy_array(np.zeros(3)) is False
    else:                                        # pragma: no cover - no CUDA
        assert fi._ensure_cupy_loaded() is True
        assert fi.cp is _optional.cupy_module()


def test_sources_core_keeps_its_cp_alias_contract():
    from lumenairy.sources import core as sc
    assert hasattr(sc, 'cp')
    assert sc.CUPY_AVAILABLE is _optional.CUPY_AVAILABLE
    if not sc.CUPY_AVAILABLE:
        assert sc._ensure_cupy_loaded() is False
        assert sc.cp is None


class TestOptionalNumba:

    def test_availability_flag_matches_find_spec(self):
        assert _optional.NUMBA_AVAILABLE is (
            importlib.util.find_spec('numba') is not None)

    def test_load_is_idempotent_and_cached(self):
        first = _optional.load_numba()
        assert first is _optional.load_numba()
        nb, njit, prange = _optional.numba_handles()
        if first:
            assert nb is not None and njit is not None and prange is not None
            nb2, njit2, prange2 = _optional.numba_handles()
            assert (nb2, njit2, prange2) == (nb, njit, prange)
        else:                                    # pragma: no cover - env
            assert (nb, njit, prange) == (None, None, None)

    def test_merit_jit_gate_is_local_so_monkeypatching_still_works(self):
        """``tests/unit/test_v5_3_multi_field_merit_jit.py`` monkeypatches
        ``_merit_jit._NUMBA_AVAILABLE = False`` to reach the pure-NumPy arm
        on a box where numba IS installed.  Delegating the IMPORT must not
        move the GATE out of the module."""
        from lumenairy.optimize import _merit_jit as mj
        assert hasattr(mj, '_NUMBA_AVAILABLE')
        args = (0.3, -0.2,
                np.linspace(0, 1, 64).reshape(8, 8),
                np.linspace(1, 2, 64).reshape(8, 8),
                np.ones((8, 8), dtype=bool))
        jitted = mj._multi_field_tilt_phasor_masked(*args, np.complex128)
        saved_flag, saved_numba = mj._NUMBA_AVAILABLE, mj._numba
        try:
            mj._NUMBA_AVAILABLE = False
            mj._numba = None
            assert mj._load_numba() is False
            plain = mj._multi_field_tilt_phasor_masked(*args, np.complex128)
        finally:
            mj._NUMBA_AVAILABLE, mj._numba = saved_flag, saved_numba
        # Same maths either way; the kernel is exp(i*(sx*kX + sy*kY)) masked.
        # Bar: 4e-16 rad, i.e. ~2 ULP of a unit-modulus complex128 phasor
        # (eps = 2.2e-16).  The two paths differ only in evaluation order of
        # a sin/cos pair, so anything above a few ULP is a real divergence.
        assert np.max(np.abs(jitted - plain)) < 4e-16


# ===========================================================================
# _knobs.py -- registry, override, snapshot / restore
# ===========================================================================

@pytest.fixture
def probe_knob():
    """A registered knob backed by a local cell, recording every setter call.

    Registered and de-registered inside the fixture so the suite's own knob
    set is untouched.
    """
    state = {'value': 'shipped', 'calls': []}

    def _get():
        return state['value']

    def _set(v):
        state['calls'].append(v)
        state['value'] = v

    _knobs.register_knob('_a15b_probe', getter=_get, setter=_set,
                         doc='test probe knob')
    try:
        yield state
    finally:
        _knobs._REGISTRY.pop('_a15b_probe', None)


@pytest.fixture
def probe_pair():
    """Two probe knobs sharing ONE call log, so restore ORDER is observable."""
    log = []
    cells = {'_a15b_a': 'a0', '_a15b_b': 'b0'}

    def _mk(name):
        def _get():
            return cells[name]

        def _set(v):
            log.append((name, v))
            cells[name] = v
        return _get, _set

    for n in ('_a15b_a', '_a15b_b'):
        g, s = _mk(n)
        _knobs.register_knob(n, getter=g, setter=s, doc=f'probe {n}')
    try:
        yield cells, log
    finally:
        for n in ('_a15b_a', '_a15b_b'):
            _knobs._REGISTRY.pop(n, None)


class TestRegistry:

    def test_every_process_global_setter_in_the_audit_is_registered(self):
        """The audit's census of genuine process-global knobs (TESTS-ARCH
        P2-5) minus the lens-family ones, which WP-A16 registers the same
        way.  A knob is visible only once its owner module is imported, so
        the imports are explicit here."""
        importlib.import_module('lumenairy.io.storage')
        importlib.import_module('lumenairy.user_library')
        importlib.import_module('lumenairy.elements.rcwa')
        registered = set(_knobs.knobs())
        expected = {
            # propagators/fft_infra.py -- the audit's 12
            'asm_cache_size', 'default_complex_dtype', 'default_dy',
            'default_real_dtype', 'default_wave_propagator',
            'fft_auto_promote', 'fft_double_buffer', 'fft_fallback',
            'fft_plan_cache_size', 'fft_plan_max_bytes_per_buffer',
            'fft_threads', 'pyfftw_planner',
            # memory.py / cache.py / io/storage.py / user_library.py
            'max_ram', 'cache_budget', 'storage_backend', 'library_path',
            # elements/rcwa/_core.py
            'blas_threads',
        }
        missing = sorted(expected - registered)
        assert not missing, (
            f'process-global knobs not registered with lumenairy._knobs: '
            f'{missing}.  Add one register_knob(...) beside the setter so '
            f'override() and the suite restore fixture reach it.')

    def test_every_registered_knob_round_trips_through_its_own_pair(self):
        """``setter(getter())`` must be a no-op for every knob -- the
        property the whole snapshot/restore machinery rests on."""
        before = _knobs.snapshot()
        for name in _knobs.knobs():
            k = _knobs._REGISTRY[name]
            k.setter(k.getter())
            assert _knobs._same(k.getter(), before[name]), (
                f'knob {name!r} does not round-trip through its own '
                f'setter/getter pair')
        assert all(_knobs._same(before[n], _knobs.snapshot()[n])
                   for n in before)

    def test_getters_are_side_effect_free_on_the_file_system(self, tmp_path):
        """``get_library_path()`` CREATES ``~/.lumenairy/library`` plus three
        subdirectories.  A snapshot runs once per test, so the registered
        getter must be the raw override instead."""
        import lumenairy.user_library as ul
        assert _knobs._REGISTRY['library_path'].getter is not ul.get_library_path
        assert _knobs.snapshot()['library_path'] == ul._library_path

    def test_registration_validates_its_arguments(self):
        with pytest.raises(ValueError, match='register_knob: name must be'):
            _knobs.register_knob('', getter=lambda: 1, setter=lambda v: None,
                                 doc='')
        with pytest.raises(TypeError, match='getter for knob'):
            _knobs.register_knob('_a15b_bad', getter=None,
                                 setter=lambda v: None, doc='')
        with pytest.raises(TypeError, match='setter for knob'):
            _knobs.register_knob('_a15b_bad', getter=lambda: 1, setter=None,
                                 doc='')
        assert '_a15b_bad' not in _knobs.knobs()

    def test_knob_doc_is_available_and_unknown_names_raise(self, probe_knob):
        assert _knobs.knob_doc('_a15b_probe') == 'test probe knob'
        assert _knobs.knob_doc('fft_threads')
        with pytest.raises(ValueError, match='knob_doc: unknown knob'):
            _knobs.knob_doc('no_such_knob')

    def test_knobs_is_sorted_and_contains_no_duplicates(self):
        names = _knobs.knobs()
        assert list(names) == sorted(names)
        assert len(set(names)) == len(names)


class TestOverride:

    def test_sets_inside_and_restores_outside(self, probe_knob):
        assert probe_knob['value'] == 'shipped'
        with la.override(_a15b_probe='scoped'):
            assert probe_knob['value'] == 'scoped'
        assert probe_knob['value'] == 'shipped'

    def test_restores_on_exception(self, probe_knob):
        with pytest.raises(RuntimeError):
            with la.override(_a15b_probe='scoped'):
                raise RuntimeError('boom')
        assert probe_knob['value'] == 'shipped'

    def test_nests_and_each_level_restores_what_it_entered_with(
            self, probe_knob):
        with la.override(_a15b_probe='outer'):
            assert probe_knob['value'] == 'outer'
            with la.override(_a15b_probe='inner'):
                assert probe_knob['value'] == 'inner'
            assert probe_knob['value'] == 'outer'
        assert probe_knob['value'] == 'shipped'

    def test_restore_order_is_the_reverse_of_entry(self, probe_pair):
        """The brief's contract: "restores the previous values in reverse
        order".  Observable here because both probes log into one list."""
        cells, log = probe_pair
        with la.override(_a15b_a='a1', _a15b_b='b1'):
            pass
        assert log == [('_a15b_a', 'a1'), ('_a15b_b', 'b1'),
                       ('_a15b_b', 'b0'), ('_a15b_a', 'a0')]

    def test_unknown_name_raises_before_anything_is_applied(self, probe_knob):
        """A typo must not leave half the block's knobs set."""
        with pytest.raises(ValueError) as exc:
            with la.override(_a15b_probe='scoped', not_a_knob=1):
                pass                              # pragma: no cover - never
        msg = str(exc.value)
        assert msg.startswith("override: unknown knob 'not_a_knob'")
        assert 'known:' in msg
        assert probe_knob['value'] == 'shipped'
        assert probe_knob['calls'] == []

    def test_a_setter_that_raises_on_entry_rolls_back(self, probe_pair):
        """Mirrors the v5.4.6 F-17 fix in ``lumenairy_context``: a mid-apply
        failure must leave the process exactly as the ``with`` found it."""
        cells, log = probe_pair

        def _boom(v):
            raise ValueError('setter refuses')

        _knobs._REGISTRY['_a15b_b'] = _knobs._REGISTRY['_a15b_b']._replace(
            setter=_boom)
        with pytest.raises(ValueError, match='setter refuses'):
            with la.override(_a15b_a='a1', _a15b_b='b1'):
                pass                              # pragma: no cover - never
        assert cells['_a15b_a'] == 'a0', (
            'the knob applied before the failing one was not rolled back')

    def test_equal_value_does_not_call_the_setter(self, probe_knob):
        """Several real setters clear caches as a side effect, so entering a
        context with the value already in force must not wipe them."""
        with la.override(_a15b_probe='shipped'):
            pass
        assert probe_knob['calls'] == []

    def test_real_knob_round_trip_through_the_public_name(self):
        from lumenairy.propagators import fft_infra as fi
        before = fi.get_fft_threads()
        with la.override(fft_threads=before + 3):
            assert fi.get_fft_threads() == before + 3
        assert fi.get_fft_threads() == before

    def test_usable_from_a_worker_thread(self):
        """Documented as PROCESS-global, not thread-local: the point of this
        pin is that calling it off the main thread works and unwinds, not
        that concurrent overrides of the SAME knob are isolated."""
        from lumenairy.propagators import fft_infra as fi
        before = fi.get_fft_threads()

        def _work(n):
            with la.override(fft_threads=n):
                return fi.get_fft_threads()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            assert ex.submit(_work, before + 5).result() == before + 5
        assert fi.get_fft_threads() == before


class TestSnapshotRestore:

    def test_restore_of_a_snapshot_is_the_identity(self):
        before = _knobs.snapshot()
        _knobs.restore(before)
        after = _knobs.snapshot()
        assert set(before) == set(after)
        for n in before:
            assert _knobs._same(before[n], after[n])

    def test_restore_undoes_a_bare_setter_call(self, probe_knob):
        """The leak this whole mechanism exists for."""
        before = _knobs.snapshot()
        probe_knob['value'] = 'leaked'            # a test that forgot finally
        assert _knobs.snapshot()['_a15b_probe'] == 'leaked'
        _knobs.restore(before)
        assert probe_knob['value'] == 'shipped'

    def test_a_knob_registered_after_the_snapshot_returns_to_its_default(
            self):
        """A module imported DURING a test registers its knob mid-flight, so
        the pre-test snapshot cannot carry it.  ``restore`` returns such a
        knob to its registration-time value, which is that module's
        import-time default -- the only value anything observed."""
        before = _knobs.snapshot()
        assert '_a15b_late' not in before
        cell = {'v': 'default'}
        _knobs.register_knob('_a15b_late', getter=lambda: cell['v'],
                             setter=lambda v: cell.__setitem__('v', v),
                             doc='late')
        try:
            cell['v'] = 'dirty'
            _knobs.restore(before)
            assert cell['v'] == 'default'
        finally:
            _knobs._REGISTRY.pop('_a15b_late', None)

    def test_restore_rejects_a_non_mapping(self):
        with pytest.raises(TypeError, match='restore: state must be'):
            _knobs.restore(['fft_threads'])

    def test_snapshot_values_are_not_aliased_for_the_mapping_knob(self):
        """``asm_cache_size`` is the one knob whose value is a dict.  Its
        getter must hand back a fresh object or a snapshot would track the
        live state instead of recording it."""
        s = _knobs.snapshot()
        assert isinstance(s['asm_cache_size'], dict)
        assert s['asm_cache_size'] is not _knobs.snapshot()['asm_cache_size']


def test_override_is_exported_at_top_level():
    assert la.override is _knobs.override
    assert 'override' in la.__all__
