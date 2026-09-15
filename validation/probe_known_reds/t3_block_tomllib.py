"""Pytest plugin -- simulate a Python 3.10 interpreter's missing ``tomllib``.

No 3.10 interpreter exists on this box (Windows: 3.13.x and 3.14.6; WSL:
3.12.3), so the 3.10 import path of
``tests/unit/test_audit2609_a15a_packaging.py`` is exercised by making
``tomllib`` unimportable in the child pytest process.

Mechanism: ``sys.modules[name] = None`` makes CPython's ``_gcd_import``
raise ``ModuleNotFoundError: import of <name> halted; None in sys.modules``
-- the SAME exception class a real 3.10 raises for a stdlib module that does
not exist, and the class the fallback under test catches.

The block is installed from the ``pytest_collection`` hook, i.e. AFTER pytest
has finished reading its own configuration out of ``pyproject.toml`` (which
it does with ``tomllib`` on 3.11+) and BEFORE any test module is imported.
Installing it at plugin-import time would break pytest itself, not the test.

Arms, via the ``T3_BLOCK`` environment variable (comma-separated):
  ``tomllib``        -- a 3.10 interpreter WITH the ``tomli`` backport
                        installed, which is what the CI ``unit`` job builds
                        (``pip install tomli``).
  ``tomllib,tomli``  -- a 3.10 interpreter with NEITHER parser.
"""
import os
import sys

_BLOCKED = [n.strip() for n in os.environ.get('T3_BLOCK', 'tomllib').split(',')
            if n.strip()]


def pytest_collection(session):
    for name in _BLOCKED:
        sys.modules[name] = None


def pytest_report_header(config):
    return ('t3_block_tomllib: python %s, will block %r at collection start'
            % (sys.version.split()[0], _BLOCKED))
