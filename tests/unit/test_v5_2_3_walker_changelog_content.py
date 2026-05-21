# v5.2.3 (AUDIT_V5_1_0 P1-NEW-2WAY-1 + V12 companion content-level walker):
"""V16 walker -- CONTENT-LEVEL CHANGELOG fabrication walker (v5.2.3).

V12 (``tests/unit/test_v5_2_walker_changelog_changeset.py``) catches
FILE-LEVEL fabrications: a backticked file-path citation that resolves
to a non-existent file, or an audit-ID citation that resolves to no
audit document.  V12 deliberately does NOT cover CONTENT-LEVEL
fabrications -- the file exists but the cited behavior is missing.
v5.1.0's ``publish.yml verify gate`` claim was that exact class: the
file existed (publish.yml has been there since v3.x); the new
``verify`` job described in the bullet was absent from the file.

This walker (V16) closes the content-level gap with the same logic as
the standalone CLI ``scripts/verify_changelog_closures.py`` -- it diffs
each audit-closure bullet's cited file paths against the
``git diff PREV_TAG..HEAD`` changeset, and asserts every code-file
citation (a ``.py`` / ``.yml`` / ``.toml`` etc. path) appears in the
diff.

The walker degrades gracefully when git history isn't available
(``pytest.skip`` rather than failing the suite on a sdist install
without a ``.git`` directory) and when the latest CHANGELOG block
has no audit-closure section to verify.

Test contract:
    test_v16_changelog_bullets_have_backing_diff
        Canonical check.  Skips on no-git / no-closures.  PASSES if
        every cited file appears in the diff.
    test_v16_companion_script_exists
        Asserts ``scripts/verify_changelog_closures.py`` is present
        and parseable.
    test_v16_companion_script_invokable
        Runs the script as a subprocess; asserts non-error exit OR
        documented-skip exit (rc == 2).
    test_v16_synthetic_fabrication_is_caught
        Constructs a synthetic CHANGELOG body in memory citing a
        non-diff path and confirms the walker would flag it.  This
        is the regression test for the walker itself.

Closes audit P1-NEW-2WAY-1 at the content level.
"""
from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Repo wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_SCRIPT = _REPO_ROOT / 'scripts' / 'verify_changelog_closures.py'


def _load_script_module():
    """Import the standalone CLI script as a module for in-process
    reuse of its parsing helpers, without requiring it on sys.path.

    Returns None if the script is missing (the caller decides how to
    react)."""
    if not _SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        '_verify_changelog_closures_v16', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_available():
    """True iff git is on PATH AND the repo has a .git directory."""
    if shutil.which('git') is None:
        return False
    return (_REPO_ROOT / '.git').exists()


# ---------------------------------------------------------------------------
# V16.1 -- canonical check
# ---------------------------------------------------------------------------

def test_v16_changelog_bullets_have_backing_diff():
    """For the topmost ``## [X.Y.Z]`` block in CHANGELOG.md, every
    audit-closure bullet's cited file path must appear in
    ``git diff PREV_TAG..HEAD``.

    Skips cleanly when:
      - git is not available (sdist install with no .git directory),
      - the latest block has no audit-closure section,
      - PREV_TAG can't be resolved (single-version CHANGELOG).
    """
    if not _git_available():
        pytest.skip(
            'git not available in this environment; V16 content-level '
            'check requires ``git diff PREV_TAG..HEAD``.')

    module = _load_script_module()
    if module is None:
        pytest.skip(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} not '
            f'present; V16 cannot verify content-level claims without '
            f'it (the walker shares parsing helpers with the script).')

    rc = module.verify_block(version=None, strict=False, quiet=True)

    if rc == 2:
        pytest.skip(
            'No audit-closure bullets in latest CHANGELOG block OR no '
            'PREV_TAG resolvable; nothing to verify.  (script exit 2)')
    if rc == 3:
        pytest.skip(
            'CHANGELOG.md not parseable by V16 walker; this is a V12 '
            'concern, not V16.  (script exit 3)')

    assert rc == 0, (
        'V16 content-level walker detected one or more audit-closure '
        'bullets whose cited file paths do NOT appear in '
        '``git diff PREV_TAG..HEAD``.  This is a v5.1.0-class '
        'CONTENT-LEVEL fabrication: the cited file may exist (V12 '
        'would not catch it) but the claimed change is not in the '
        'diff.  Re-run the standalone script for detail:\n\n'
        '    python scripts/verify_changelog_closures.py\n\n'
        'Either ship the missing change, remove the bullet, or mark '
        'it ``[NOT SHIPPED -- moved to v<next>]`` per the v5.1.1 '
        'retroactive-tagging precedent.')


# ---------------------------------------------------------------------------
# V16.2 -- companion script presence + parseability
# ---------------------------------------------------------------------------

def test_v16_companion_script_exists():
    """``scripts/verify_changelog_closures.py`` must be present and
    parse as valid Python.  V12.4 also asserts existence as an
    OR-condition with other artifacts; V16.2 makes the script a hard
    requirement of v5.2.3."""
    assert _SCRIPT.is_file(), (
        f'expected companion script at '
        f'{_SCRIPT.relative_to(_REPO_ROOT)}; v5.2.3 ROADMAP closure '
        f'requires both the V16 walker AND the standalone script.  '
        f'See CHANGELOG.md ``## [5.2.3]`` block + ROADMAP v5.2.')

    source = _SCRIPT.read_text(encoding='utf-8')
    try:
        ast.parse(source, filename=str(_SCRIPT))
    except SyntaxError as exc:
        pytest.fail(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} does '
            f'not parse as Python: {exc!r}')

    # Sanity: docstring + CLI entry point
    assert 'verify_block' in source, (
        f'companion script missing the ``verify_block`` entry point '
        f'that V16 walker shares for in-process reuse.')
    assert '--version' in source, (
        f'companion script missing the documented ``--version`` flag.')
    assert '--strict' in source, (
        f'companion script missing the documented ``--strict`` flag.')


# ---------------------------------------------------------------------------
# V16.3 -- companion script invokable as a subprocess
# ---------------------------------------------------------------------------

def test_v16_companion_script_invokable():
    """Run the script as a subprocess against a known-good past
    version (v5.2.0 has real backing diffs) and assert it either
    succeeds (rc 0) or skips cleanly (rc 2 -- no git history).

    Anything else is a regression in the script's CLI surface.
    """
    if not _SCRIPT.is_file():
        pytest.skip(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} not '
            f'present; V16.3 cannot exercise CLI surface.')

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--version', '5.2.0', '--quiet'],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode in (0, 2), (
        f'companion script exited with rc={result.returncode} on a '
        f'past version that should have real backing diffs.\n'
        f'stdout: {result.stdout!r}\n'
        f'stderr: {result.stderr!r}')


# ---------------------------------------------------------------------------
# V16.4 -- self-test: walker actually catches a synthetic fabrication
# ---------------------------------------------------------------------------

def test_v16_synthetic_fabrication_is_caught(tmp_path, monkeypatch):
    """Regression test for the walker itself.  Constructs a synthetic
    CHANGELOG.md with an audit-closure bullet citing a path that is
    not in any real diff, and confirms the walker flags it.

    Without this self-test, a refactor that accidentally turned the
    walker into a no-op would silently pass the rest of the suite.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            f'companion script not present; V16.4 requires the script '
            f'module for in-process parsing helpers.')

    if not _git_available():
        pytest.skip(
            'git not available; V16.4 needs a real diff range to '
            'compare against.')

    # Compose a synthetic CHANGELOG with one fabricated audit-closure
    # bullet citing a deliberately nonexistent file.  We embed two
    # version blocks so the walker's _select_block can identify a
    # PREV_TAG candidate (which here must be a real existing tag for
    # the diff command to succeed -- we use v5.1.1 as that target).
    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [5.1.1] -- 2026-05-20\n\n'
        '### Audit closures\n\n'
        '* **Fabricated closure** at '
        '`lumenairy/nonexistent_file_for_v16_test.py:42-51` -- this '
        'bullet cites a file that is NOT in any real v5.1.0..v5.1.1 '
        'diff.  V16 walker must flag it.\n\n'
        '## [5.1.0] -- 2026-05-20\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )

    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    # Point the script module at the synthetic CHANGELOG without
    # touching the real repo file.  ``_REPO_ROOT`` is used by the git
    # plumbing -- we leave it alone because the synthetic block's
    # PREV_TAG (v5.1.0) and HEAD-side ref (v5.1.1) ARE real tags, so
    # the diff command itself succeeds; only the cited path is
    # fabricated.
    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)

    rc = module.verify_block(version='5.1.1', strict=False, quiet=True)
    assert rc == 1, (
        f'V16 walker failed to flag a synthetic fabrication.  The '
        f'cited path ``lumenairy/nonexistent_file_for_v16_test.py`` '
        f'is not in the v5.1.0..v5.1.1 diff and should have produced '
        f'exit code 1; got rc={rc} instead.  This means the walker '
        f'is silently passing fabrications -- a regression in the '
        f'V16 contract itself.')
