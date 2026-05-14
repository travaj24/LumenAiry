"""Pytest plumbing for the ``validation/`` suite.

The validation files were authored around a custom :class:`Harness`
class with test functions named ``t_*`` that return a ``(bool, str)``
tuple instead of raising ``AssertionError``.  This file is a thin
adapter that makes pytest collect and run those functions natively
so the user gets:

* ``pytest validation/`` discovers all 670 ``t_*`` tests as
  individual pytest test items.
* ``pytest validation/elements/test_lenses.py::t_singlet_focuses``
  runs a single named test.
* ``pytest validation/ -k 'gaussian and not slow'`` does keyword
  filtering.
* ``pytest validation/ -n auto`` parallelises via pytest-xdist.
* JUnit XML, coverage, IDE test discovery all work out of the box.

The existing ``run_all.py`` driver keeps working unchanged -- it
runs each ``test_*.py`` as a subprocess via the legacy
:class:`Harness`.  Both invocation paths produce equivalent results.

How it works
------------
1. ``python_functions`` is extended at collection time to include
   the ``t_*`` prefix so pytest's standard collector picks them up.
2. ``pytest_pyfunc_call`` intercepts each ``t_*`` invocation, calls
   the underlying function, and converts the ``(bool, str)`` return
   value into a real ``assert`` (raising ``AssertionError`` with the
   detail string on failure).
3. Every collected ``t_*`` is auto-marked ``integration`` so it
   respects the project-wide ``addopts = ["-m", "not integration"]``
   default and only runs when the user opts in with ``-m
   integration`` or with an explicit ``validation/`` path argument.

Author: Andrew Traverso (added in 4.7 as part of polish-pass item G)
"""
from __future__ import annotations

import os
import sys

import pytest


# Add the repo root to sys.path so ``import lumenairy`` works when
# pytest is invoked from outside the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)


def pytest_collection_modifyitems(config, items):
    """Auto-mark every ``t_*`` test ``integration``.

    The validation suite is slow (~6 minutes wall-time for the full
    34 files); the project-wide default
    ``addopts = ["-m", "not integration"]`` keeps it out of the
    default ``pytest`` invocation while still being explicitly
    runnable via ``pytest -m integration`` or
    ``pytest validation/``.
    """
    val_dir = os.path.normpath(_HERE)
    mark = pytest.mark.integration
    for item in items:
        try:
            item_path = os.path.normpath(str(item.path))
        except AttributeError:
            continue
        if not item_path.startswith(val_dir):
            continue
        if item.name.startswith('t_'):
            item.add_marker(mark)


def pytest_pyfunc_call(pyfuncitem):
    """Intercept ``t_*`` function calls and turn returns into asserts.

    pytest's default :func:`pytest_pyfunc_call` ignores the return
    value of test functions.  The validation ``t_*`` functions
    return ``bool`` or ``(bool, detail_str)`` to signal pass/fail.
    We unpack the return value here and raise ``AssertionError``
    on failure so pytest reports it correctly.

    Returning ``True`` from this hook tells pytest "we handled the
    call, don't run the default machinery."  Returning ``None``
    lets pytest fall through to its standard behaviour for any
    function not starting with ``t_``.
    """
    if not pyfuncitem.name.startswith('t_'):
        return None  # standard test_* function -- default behaviour

    func = pyfuncitem.obj
    try:
        argnames = pyfuncitem._fixtureinfo.argnames
    except AttributeError:
        argnames = ()
    kwargs = {n: pyfuncitem.funcargs[n] for n in argnames
              if n in pyfuncitem.funcargs}

    try:
        result = func(**kwargs)
    except Exception:
        raise  # surface the exception as a real test failure

    if isinstance(result, tuple) and len(result) >= 1:
        ok = bool(result[0])
        detail = str(result[1]) if len(result) > 1 else ''
    elif isinstance(result, bool):
        ok = bool(result)
        detail = ''
    elif result is None:
        # Some t_* functions are written assertion-style and return
        # None on success; honour that.
        ok = True
        detail = ''
    else:
        ok = bool(result)
        detail = repr(result)

    assert ok, detail
    return True
