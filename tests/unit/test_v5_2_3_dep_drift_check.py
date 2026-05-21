"""v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
smoke tests for ``scripts/check_dep_metadata.py``.

The script is a standalone PyPI-metadata drift watcher that lives at
``scripts/check_dep_metadata.py`` and is invoked from
``.github/workflows/dep-drift.yml`` on a weekly cron.  See the script's
module docstring for the v5.2.2 retrospective (pyfftw 0.15.1 / zarr
3.1.6 dropping Python 3.10 support).

These smoke tests assert:

1. The script file exists and parses as valid Python.
2. ``python scripts/check_dep_metadata.py --help`` exits 0 cleanly.
3. The pyproject parser inside the script reads the current
   ``pyproject.toml`` without error.

The tests do NOT make any network calls to PyPI -- that's exercised by
the weekly workflow, not by the PR-gating unit suite.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_dep_metadata.py"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


# ----------------------------------------------------------------------
# Test 1: file exists and parses
# ----------------------------------------------------------------------


def test_script_exists_and_parses() -> None:
    """v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
    ``scripts/check_dep_metadata.py`` exists and is syntactically valid
    Python.  Catches the trivial "script was deleted" / "merge conflict
    left a syntax error" regression.
    """
    assert _SCRIPT_PATH.is_file(), (
        f"v5.2.3 drift-check script missing at {_SCRIPT_PATH}.  "
        "This file is the target of the weekly dep-drift workflow "
        "(.github/workflows/dep-drift.yml); both must ship together."
    )

    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(_SCRIPT_PATH))
    except SyntaxError as e:  # pragma: no cover -- diagnostic only
        pytest.fail(
            f"v5.2.3 drift-check script does not parse: {e}.  "
            "The weekly workflow will fail every Monday until this is fixed."
        )


# ----------------------------------------------------------------------
# Test 2: --help exits cleanly
# ----------------------------------------------------------------------


def test_script_help_exits_cleanly() -> None:
    """v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
    ``python scripts/check_dep_metadata.py --help`` exits 0 and prints
    a description containing the script's purpose.  Catches argparse
    misconfiguration that would break the workflow's first-line sanity.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"v5.2.3 drift-check --help exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The argparse help should mention the script's purpose -- a quick
    # sanity check that the docstring / description didn't get stripped.
    assert "drift" in result.stdout.lower(), (
        "v5.2.3 drift-check --help does not contain the word 'drift'; "
        "the argparse description appears to be missing or wrong:\n"
        f"{result.stdout}"
    )


# ----------------------------------------------------------------------
# Test 3: pyproject parser reads the current pyproject.toml
# ----------------------------------------------------------------------


def _load_script_module():
    """Import the script as a module so we can call its parser directly.

    The script is a top-level ``.py`` file in ``scripts/``, not part of
    a package, so we use ``importlib.util.spec_from_file_location``
    rather than a plain ``import``.  This is the same idiom used in
    other v5.2.x verification helpers.
    """
    spec = importlib.util.spec_from_file_location(
        "_v5_2_3_dep_drift_check", _SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover -- defensive
        pytest.skip(
            f"Could not build import spec for {_SCRIPT_PATH}; "
            "skipping pyproject-parse smoke test."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_parser_reads_current_pyproject() -> None:
    """v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
    The script's ``parse_pyproject_deps`` function reads the current
    repo's ``pyproject.toml`` without error and returns a non-empty
    mapping that includes the well-known hard deps (numpy, scipy) and
    at least one optional extra (e.g. pyfftw from the v5.2.2 fix).

    This is the load-bearing assertion: if the script ever loses the
    ability to parse our pyproject, the weekly workflow gives a false-
    OK and the v5.2.2-class regression goes undetected.
    """
    assert _PYPROJECT.is_file(), f"pyproject.toml missing at {_PYPROJECT}"

    module = _load_script_module()
    deps = module.parse_pyproject_deps(_PYPROJECT)

    assert isinstance(deps, dict) and deps, (
        "parse_pyproject_deps returned an empty / non-dict result for "
        "the current repo's pyproject.toml; the parser is broken."
    )

    # Hard deps from pyproject.toml [project.dependencies].
    for must_have in ("numpy", "scipy", "matplotlib", "psutil"):
        assert must_have in deps, (
            f"v5.2.3 drift-check parser missed hard dep {must_have!r}; "
            f"observed keys: {sorted(deps)}"
        )

    # At least one optional dep should show up too -- pyfftw is the
    # v5.2.2 incident centrepiece, so it's a particularly meaningful
    # canary that the optional-extras walker is functioning.
    assert "pyfftw" in deps, (
        "v5.2.3 drift-check parser missed optional dep 'pyfftw' "
        "(the v5.2.2 incident centrepiece); the optional-extras walker "
        f"is broken.  Observed keys: {sorted(deps)}"
    )
