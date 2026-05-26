"""Pytest wrapper around the validation/ harness (integration suite).

Discovers each ``validation/*/test_*.py`` file and runs it as a
parametrized pytest test, capturing the per-file pass/fail.  This
gives pytest's UX (``-x``, ``-k``, ``-n auto`` via pytest-xdist,
JUnit XML, coverage hooks) without refactoring the existing
``_harness.Harness`` infrastructure that the validation files
rely on.

Run with:

    pytest tests/integration/ -v             # standard
    pytest tests/integration/ -n auto        # parallel (pytest-xdist)
    pytest tests/integration/ -k 'asymptotic'  # filter

In 4.3.0 this was relocated from ``tests/`` to ``tests/integration/``
so it sits alongside the fast ``tests/unit/`` suite that is the
default ``pytest`` target.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
VALIDATION = REPO / 'validation'


def _discover_validation_files():
    """Return a list of all ``validation/<subdir>/test_*.py`` files."""
    files = []
    for sub in sorted(VALIDATION.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith('_') or sub.name.startswith('.'):
            continue
        for f in sorted(sub.glob('test_*.py')):
            files.append(f)
    for f in sorted(VALIDATION.glob('test_*.py')):
        files.append(f)
    return files


_VAL_FILES = _discover_validation_files()
_VAL_IDS = [f.relative_to(VALIDATION).as_posix() for f in _VAL_FILES]


@pytest.mark.integration
@pytest.mark.parametrize('val_file', _VAL_FILES, ids=_VAL_IDS)
def test_validation_file_passes(val_file):
    """Run ``python <val_file>`` in a subprocess and assert exit 0."""
    proc = subprocess.run(
        [sys.executable, str(val_file)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"\n{val_file.name} FAILED (exit {proc.returncode})\n\n"
            f"--- stdout (last 2000 chars) ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{proc.stderr[-2000:]}\n"
        )
