"""Pytest wrapper around the validation/ harness.

Discovers each ``validation/*/test_*.py`` file and runs it as a
parametrized pytest test, capturing the per-file pass/fail.  This
gives pytest's UX (``-x``, ``-k``, ``-n auto`` via pytest-xdist,
JUnit XML, coverage hooks) without refactoring the existing
``_harness.Harness`` infrastructure that the validation files
rely on.

Run with:

    pytest tests/test_validation_files.py -v             # standard
    pytest tests/test_validation_files.py -n auto        # parallel
    pytest tests/test_validation_files.py -k 'asymptotic'  # filter
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
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
    # Top-level validation files
    for f in sorted(VALIDATION.glob('test_*.py')):
        files.append(f)
    return files


_VAL_FILES = _discover_validation_files()
_VAL_IDS = [f.relative_to(VALIDATION).as_posix() for f in _VAL_FILES]


@pytest.mark.parametrize('val_file', _VAL_FILES, ids=_VAL_IDS)
def test_validation_file_passes(val_file):
    """Run ``python <val_file>`` in a subprocess and assert exit 0.

    The subprocess approach matches what ``validation/run_all.py``
    does, but parametrization gives pytest's per-test isolation,
    fail-fast (``-x``), filtering (``-k``), and parallel execution
    (``-n auto`` via pytest-xdist).
    """
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
