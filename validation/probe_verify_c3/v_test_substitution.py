"""VERIFY-WP-C3 CLAIM 8f -- does the substituted-module test reach the True
side on a build with NO CuPy, and does it leave the module clean?

Run with pytest, from the tree under test, in ONE process::

    python -m pytest validation/probe_verify_c3/v_test_substitution.py -q

Three tests in file order: the state BEFORE, the shipped test run through a
real ``MonkeyPatch``, and the state AFTER.  ``test_3`` fails on any leak.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import numpy as np
import pytest

from lumenairy.propagators import fft_infra as fi

_BEFORE = {}
_REACHED = {}


def _snapshot():
    return {
        'cp': fi.cp,
        'CUPY_AVAILABLE': fi.CUPY_AVAILABLE,
        '_optional_is_cupy_array': fi._optional_is_cupy_array,
        '_ensure_cupy': fi._ensure_cupy,
        'is_cupy_of_numpy': fi._is_cupy_array(np.zeros(3)),
    }


def _load_shipped():
    root = pathlib.Path(fi.__file__).parents[2]
    p = root / 'tests' / 'unit' / 'test_c3_collins_default.py'
    assert p.is_file(), p
    spec = importlib.util.spec_from_file_location('shipped_c3', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_1_state_before():
    _BEFORE.update(_snapshot())
    try:
        import cupy  # noqa: F401
        _BEFORE['cupy_importable'] = True
    except Exception:                                 # noqa: BLE001
        _BEFORE['cupy_importable'] = False
    print('\nBEFORE: cp=%r CUPY_AVAILABLE=%r cupy_importable=%r'
          % (_BEFORE['cp'], _BEFORE['CUPY_AVAILABLE'],
             _BEFORE['cupy_importable']))


def test_2_run_the_shipped_substitution_test():
    """Run the shipped test with a REAL MonkeyPatch, and independently record
    that its True side was reached -- by wrapping ``_is_cupy_array`` so the
    answer it got is observed from outside the test."""
    mod = _load_shipped()
    fn = mod.test_a_true_cupy_answer_really_binds_the_fft_dispatchers_cp
    mp = pytest.MonkeyPatch()
    real = fi._is_cupy_array
    seen = []

    def spy(x):
        r = real(x)
        seen.append((type(x).__name__, r, fi.cp is not None))
        return r
    fi._is_cupy_array = spy
    try:
        fn(mp)
    finally:
        fi._is_cupy_array = real
        mp.undo()
    _REACHED['calls'] = list(seen)
    print('\n_is_cupy_array answers observed during the shipped test: %r'
          % seen)
    assert any(r is True for _t, r, _b in seen), (
        'the shipped test never got a True answer out of _is_cupy_array, so '
        'it did not exercise the True side on this build')
    assert any(r is True and bound for _t, r, bound in seen), (
        'a True answer was given while the module-level cp was still unbound')


def test_3_state_after_is_clean():
    after = _snapshot()
    leaks = {k: (_BEFORE[k], after[k]) for k in after
             if _BEFORE.get(k) is not after[k]
             and _BEFORE.get(k) != after[k]}
    print('\nAFTER : cp=%r CUPY_AVAILABLE=%r' % (after['cp'],
                                                 after['CUPY_AVAILABLE']))
    assert not leaks, 'fft_infra was left mutated by the shipped test: %r' \
        % leaks
    # and a real NumPy array is still answered False through the real chain
    assert fi._is_cupy_array(np.zeros(3)) is False
    # the stub must not be reachable any more
    assert fi.cp is None or type(fi.cp).__name__ != 'module' or \
        getattr(fi.cp, '__name__', '') != 'cupy_stub', fi.cp


def test_4_the_dispatcher_still_transforms():
    x = np.arange(16, dtype=np.complex128).reshape(4, 4)
    a = fi._fft2(np.ascontiguousarray(x))
    assert np.allclose(np.asarray(a), np.fft.fft2(x))
    print('\nos.cpu_count=%s' % os.cpu_count())
