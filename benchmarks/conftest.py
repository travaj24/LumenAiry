"""Benchmark suite for LumenAiry v4.12+.

Each ``test_bench_*.py`` file is a pytest-benchmark file targeting one
specific code path.  Files are NOT collected by the regular unit suite
(see ``pytest.ini`` / ``pyproject.toml`` ``testpaths``).
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark every benchmark item with the ``bench`` mark so callers can
    select / deselect via ``-m bench`` / ``-m 'not bench'``."""
    for item in items:
        item.add_marker(pytest.mark.bench)
