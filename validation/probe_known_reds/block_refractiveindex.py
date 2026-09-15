"""Make the optional ``refractiveindex`` package UNIMPORTABLE in this process.

Why this exists
---------------
CI deliberately does not install the glass extra (``.github/workflows/
unit-tests.yml``, install step: ``[all,dev]`` pulled jax + astropy +
refractiveindex and was dropped on purpose, with ``validate.yml`` named as the
canonical harness for those paths).  "refractiveindex is absent" is therefore a
SUPPORTED configuration, and the A8/E2 reds of CI run 34914295323 only appear
there.  This box HAS the package installed, so the absence has to be a FIXTURE
rather than an accident of the machine.

Fidelity
--------
``lumenairy.glass`` decides availability exactly once, at import, with
``importlib.util.find_spec('refractiveindex') is not None``, and loads the class
lazily with ``from refractiveindex import RefractiveIndexMaterial``.  Blocking
both of those reproduces the CI-observable state exactly:

* ``find_spec`` returns ``None``  -> ``_REFRACTIVEINDEX_AVAILABLE is False``
  (same value a real absence gives);
* ``sys.modules['refractiveindex'] = None`` -> any surviving ``import
  refractiveindex`` raises ``ImportError``, as a real absence would.

``install()`` MUST run before ``lumenairy.glass`` is first imported; it asserts
that it was not imported already, so a mis-ordered caller fails loudly instead
of silently measuring the package-present arm.

Usage::

    python -c "import sys; sys.path.insert(0, 'validation/probe_known_reds'); \
               import block_refractiveindex as b; b.install(); \
               import pytest; sys.exit(pytest.main([...]))"
"""
from __future__ import annotations

import importlib.util
import sys

_NAME = 'refractiveindex'
_real_find_spec = importlib.util.find_spec


def _blocked_find_spec(name, package=None):
    if name == _NAME or name.startswith(_NAME + '.'):
        return None
    return _real_find_spec(name, package)


def install():
    """Block the package.  Raises if lumenairy.glass already resolved it."""
    assert 'lumenairy.glass' not in sys.modules, (
        'block_refractiveindex.install() must run BEFORE lumenairy.glass is '
        'imported -- it snapshots availability at import time')
    importlib.util.find_spec = _blocked_find_spec
    # ``None`` in sys.modules is the documented "this import is halted" marker;
    # it makes ``import refractiveindex`` raise ImportError.
    sys.modules[_NAME] = None
    for mod in [m for m in sys.modules if m.startswith(_NAME + '.')]:
        sys.modules[mod] = None
