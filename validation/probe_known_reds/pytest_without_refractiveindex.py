"""Run pytest with the optional ``refractiveindex`` package blocked.

Reproduces CI's install configuration (``.github/workflows/unit-tests.yml``
deliberately does NOT install the glass extra) on a box where the package IS
present.  ``block_refractiveindex.install()`` runs before pytest imports any
test module, so ``lumenairy.glass`` snapshots availability as ``False`` exactly
as it does on a runner with no package installed.

Usage::

    python validation/probe_known_reds/pytest_without_refractiveindex.py \
        tests/unit/test_audit2609_a8_glass.py --capture=sys -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_refractiveindex                                   # noqa: E402

block_refractiveindex.install()

import pytest                                                  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
