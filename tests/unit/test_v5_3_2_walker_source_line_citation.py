# v5.3.2 (ROADMAP V18 walker -- source-file:line citation drift):
"""V18 walker -- SOURCE-FILE:LINE CHANGELOG citation drift walker (v5.3.2).

V12 (``tests/unit/test_v5_2_walker_changelog_changeset.py``) catches
FILE-LEVEL fabrications: a backticked file-path citation that resolves
to a non-existent file.  V16
(``tests/unit/test_v5_2_3_walker_changelog_content.py``) catches
CONTENT-LEVEL fabrications: the file exists but the cited change is
absent from the ``git diff PREV_TAG..HEAD`` changeset.  V17
(``tests/unit/test_v5_3_walker_changelog_self_citation.py``) catches
RECURSIVE SELF-CITATION drift in the CHANGELOG's own test-/file-/
line-count headline numbers.

V18 closes a fourth sibling: LINE-LEVEL DRIFT in source-file:line
citations.

The v5.3.0 ship hit a real instance.  v5.3.0's CHANGELOG cited
``optimize/wrapper_merits.py:855`` for the ``_ZERO_APERTURE_MASK``
sentinel branch.  The v5.3.0 MultiFieldMerit JIT change added ~19
LOC ABOVE the sentinel branch in the same file, shifting the cited
site to line 876.  The ``test_v4_15_agent_f.py::TestF5ChangelogLineCitations``
pin caught the drift but only for a few hard-coded symbols it knows
about (it builds an anchor-string database BY HAND for the specific
sites the v4.14.2 audit cited).

V18 is the GENERAL version of that pin: it parses CHANGELOG.md for
ALL backticked ``lumenairy/foo/bar.py:N`` (and bare-basename
``mhs.py:N``) citations in the topmost block, opens each cited file
at the cited line, and verifies the line is "non-trivial" -- NOT
just whitespace, NOT a closing bracket, NOT a bare
``pass``/``continue``/``break``.  A non-trivial line at the cited
position is the structural anchor; a trivial line means the cited
symbol has almost certainly drifted.

This is a NECESSARY-BUT-NOT-SUFFICIENT check.  V18 does not try to
verify the cited SYMBOL is at the cited line (that requires the
anchor-string database that ``test_v4_15_agent_f.py`` builds by hand
for specific symbols).  V18 just verifies "the cited line still
exists AND is non-trivial."

Test contract:
    test_v18_source_line_citations_point_at_non_trivial_lines
        Canonical check.  Skips on no-citations.  PASSES if every
        cited line is non-trivial.
    test_v18_companion_script_exists
        Asserts ``scripts/check_source_line_citations.py`` is present
        and parseable.
    test_v18_companion_script_invokable
        Runs the script as a subprocess; asserts non-error exit OR
        documented-skip exit (rc == 2).
    test_v18_walker_synthetic_stale_citation_is_caught
        Constructs a synthetic CHANGELOG with a citation pointing
        at a deliberately blank line; asserts the walker would flag
        it.  Self-test for the walker itself.

Closes ROADMAP v5.3.2 item #2 (V18 walker).
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_SCRIPT = _REPO_ROOT / 'scripts' / 'check_source_line_citations.py'


def _load_script_module():
    """Import the standalone CLI script as a module for in-process
    reuse of its parsing helpers, without requiring it on sys.path.

    Returns None if the script is missing (the caller decides how
    to react).
    """
    if not _SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        '_check_source_line_citations_v18', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# V18.1 -- canonical check: cited lines are non-trivial
# ===========================================================================

def test_v18_source_line_citations_point_at_non_trivial_lines():
    """Every source-file:line citation in the topmost CHANGELOG
    ``## [X.Y.Z]`` block must land on a non-trivial line.

    A non-trivial line is one that is NOT just whitespace, NOT a
    closing bracket, and NOT a bare ``pass`` / ``continue`` /
    ``break`` keyword.  The cited line acts as the structural anchor
    for the bullet's claim; if it's trivial, the cited symbol has
    drifted.

    Skips cleanly when the block has no source-file:line citations
    (e.g. a docs-only patch release like v5.3.1).
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} not '
            f'present; V18 cannot verify line-level citations without '
            f'the shared parsing helpers.')

    rc = module.audit_block(version=None, quiet=True)

    if rc == 2:
        pytest.skip(
            'No source-file:line citations in latest CHANGELOG block; '
            'nothing to verify.  (script exit 2)')

    assert rc == 0, (
        'V18 line-level walker detected one or more source-file:line '
        'citations whose cited line is TRIVIAL (whitespace, closing '
        'bracket, or bare ``pass``/``continue``/``break``).  This is a '
        'v5.3.0-class CITATION DRIFT: the cited file exists (V12 '
        'would not catch it) and the bullet may have a backing diff '
        '(V16 would not catch it) but the cited LINE NUMBER has '
        'drifted from the actual symbol location.  Re-run the '
        'standalone script for detail:\n\n'
        '    python scripts/check_source_line_citations.py\n\n'
        'Either refresh the line citation to the current line number '
        'or replace ``:N`` with a stable anchor (the symbol name) so '
        'the citation is line-independent.')


# ===========================================================================
# V18.2 -- companion script presence + parseability
# ===========================================================================

def test_v18_companion_script_exists():
    """``scripts/check_source_line_citations.py`` must be present and
    parse as valid Python.

    Parallel to V16.2's check for ``verify_changelog_closures.py``.
    v5.3.2 ROADMAP closure requires both the V18 walker AND the
    standalone script.
    """
    assert _SCRIPT.is_file(), (
        f'expected companion script at '
        f'{_SCRIPT.relative_to(_REPO_ROOT)}; v5.3.2 ROADMAP closure '
        f'requires both the V18 walker AND the standalone script.')

    source = _SCRIPT.read_text(encoding='utf-8')
    try:
        ast.parse(source, filename=str(_SCRIPT))
    except SyntaxError as exc:
        pytest.fail(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} does '
            f'not parse as Python: {exc!r}')

    # Sanity: docstring + CLI entry point names V18 walker shares.
    assert 'audit_block' in source, (
        'companion script missing the ``audit_block`` entry point '
        'that V18 walker shares for in-process reuse.')
    assert '--version' in source, (
        'companion script missing the documented ``--version`` flag.')
    assert '--quiet' in source, (
        'companion script missing the documented ``--quiet`` flag.')


# ===========================================================================
# V18.3 -- companion script invokable as a subprocess
# ===========================================================================

def test_v18_companion_script_invokable():
    """Run the script as a subprocess against the topmost CHANGELOG
    block; assert it either succeeds (rc 0) or skips cleanly (rc 2,
    no citations).

    Anything else is a regression in the script's CLI surface.
    """
    if not _SCRIPT.is_file():
        pytest.skip(
            f'companion script {_SCRIPT.relative_to(_REPO_ROOT)} not '
            f'present; V18.3 cannot exercise CLI surface.')

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--quiet'],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode in (0, 2), (
        f'companion script exited with rc={result.returncode} on the '
        f'topmost CHANGELOG block.  Expected 0 (OK) or 2 (skip-clean, '
        f'no citations).\n'
        f'stdout: {result.stdout!r}\n'
        f'stderr: {result.stderr!r}')


# ===========================================================================
# V18.4 -- self-test: walker actually catches a synthetic stale citation
# ===========================================================================

def test_v18_walker_synthetic_stale_citation_is_caught(tmp_path, monkeypatch):
    """Regression test for the walker itself.  Constructs a synthetic
    CHANGELOG.md citing a file:line that points at a deliberately
    BLANK line, and confirms the walker flags it.

    Without this self-test, a refactor that accidentally turned the
    walker into a no-op (e.g. ``_is_trivial_line`` always returning
    False) would silently pass the rest of the suite.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; V18.4 requires the script '
            'module for in-process parsing helpers.')

    # Create a real source file under tmp_path with a known-trivial
    # line at a known position.  Line 1 has content; line 2 is blank.
    # Line 3 has content.  The synthetic CHANGELOG cites line 2.
    fake_pkg = tmp_path / 'lumenairy'
    fake_pkg.mkdir()
    fake_src = fake_pkg / 'fake_module_for_v18_test.py'
    fake_src.write_text(
        'x = 1\n'   # line 1 -- non-trivial
        '\n'        # line 2 -- BLANK (trivial)
        '\n'        # line 3 -- BLANK (trivial)
        '\n'        # line 4 -- BLANK (trivial)
        '\n'        # line 5 -- BLANK (trivial)
        '\n',       # line 6 -- BLANK (trivial)
        encoding='utf-8',
    )

    # Synthetic CHANGELOG body cites the blank line.  Two version
    # blocks so the walker's _select_block has something to pick.
    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [9.9.9] -- 2099-12-31\n\n'
        '### What changed\n\n'
        '* **Fabricated line citation** at '
        '`lumenairy/fake_module_for_v18_test.py:3` -- this bullet '
        'cites line 3, which is a BLANK line (trivial).  V18 walker '
        'must flag it.\n\n'
        '## [9.9.8] -- 2099-12-30\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )
    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    # Point the script module at the synthetic CHANGELOG + tmp_path
    # as repo root so the basename / full-path resolver finds the
    # fake source file.  The basename index is cached on first call,
    # so we also clear it.
    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)
    monkeypatch.setattr(module, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(module, '_BASENAME_INDEX_CACHE', None)

    rc = module.audit_block(version='9.9.9', quiet=True)
    assert rc == 1, (
        f'V18 walker failed to flag a synthetic stale citation.  The '
        f'cited line ``lumenairy/fake_module_for_v18_test.py:3`` is a '
        f'BLANK line and should have produced exit code 1; got rc={rc} '
        f'instead.  This means the walker is silently passing line-'
        f'level drift -- a regression in the V18 contract itself.')


# ===========================================================================
# V18.5 -- self-test: walker correctly PASSES a non-trivial citation
# ===========================================================================

def test_v18_walker_synthetic_valid_citation_is_passed(tmp_path, monkeypatch):
    """Companion to V18.4: confirm the walker does NOT false-positive
    on a valid citation.  Cites a non-trivial line in a synthetic
    file; walker should return rc=0.

    Defends against a future ``_is_trivial_line`` regression that
    would mark every line as trivial (i.e. always-fail the walker).
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; V18.5 requires the script '
            'module for in-process parsing helpers.')

    fake_pkg = tmp_path / 'lumenairy'
    fake_pkg.mkdir()
    fake_src = fake_pkg / 'valid_module_for_v18_test.py'
    fake_src.write_text(
        'def hello():\n'           # line 1 -- non-trivial
        '    return "world"\n'      # line 2 -- non-trivial
        '\n'                        # line 3 -- blank
        'CONSTANT = 42\n',          # line 4 -- non-trivial
        encoding='utf-8',
    )

    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [9.9.9] -- 2099-12-31\n\n'
        '### What changed\n\n'
        '* **Valid line citation** at '
        '`lumenairy/valid_module_for_v18_test.py:1` -- line 1 has '
        'the ``def hello`` symbol; V18 walker must accept it.\n'
        '* Another at `lumenairy/valid_module_for_v18_test.py:4` -- '
        'line 4 has ``CONSTANT = 42``; also accepted.\n\n'
        '## [9.9.8] -- 2099-12-30\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )
    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)
    monkeypatch.setattr(module, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(module, '_BASENAME_INDEX_CACHE', None)

    rc = module.audit_block(version='9.9.9', quiet=True)
    assert rc == 0, (
        f'V18 walker false-positived on a valid citation.  Both cited '
        f'lines (1 and 4) are non-trivial and the walker should have '
        f'returned 0; got rc={rc} instead.  This means the walker is '
        f'rejecting valid citations -- a regression in the V18 '
        f'trivial-line heuristic.')


# ===========================================================================
# v5.4 hardening tests -- AUDIT_V5_3_2 Part 7 P2 closures
# ===========================================================================

def test_v18_ambiguous_basename_contributes_to_drift(tmp_path, monkeypatch):
    """v5.4 (audit V17/V18 hardening) -- AUDIT_V5_3_2 Part 7 P2-1:
    bare-basename citations resolving to MULTIPLE candidate files now
    contribute to drift (rc=1) rather than being silently skipped.

    Constructs two files with the same basename in different
    subtrees, then a synthetic CHANGELOG citing the bare basename.
    The walker must return rc=1 with a WARN message naming the
    competing candidates.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; v5.4 ambiguous-basename '
            'test requires the script module.')

    # Two files with the same basename in different subtrees.  The
    # walker's basename index searches lumenairy/, tests/, scripts/,
    # examples/, benchmarks/, validation/.  Build two real candidates.
    lumenairy_pkg = tmp_path / 'lumenairy'
    lumenairy_pkg.mkdir()
    (lumenairy_pkg / 'duplicate_basename.py').write_text(
        'x = 1\n', encoding='utf-8')
    tests_pkg = tmp_path / 'tests'
    tests_pkg.mkdir()
    (tests_pkg / 'duplicate_basename.py').write_text(
        'y = 2\n', encoding='utf-8')

    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [9.9.9] -- 2099-12-31\n\n'
        '### What changed\n\n'
        '* **Ambiguous basename citation** at '
        '`duplicate_basename.py:1` -- this bullet cites a bare '
        'basename that matches multiple files.  V18 walker must '
        'flag it as drift.\n\n'
        '## [9.9.8] -- 2099-12-30\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )
    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)
    monkeypatch.setattr(module, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(module, '_BASENAME_INDEX_CACHE', None)

    rc = module.audit_block(version='9.9.9', quiet=True)
    assert rc == 1, (
        f'V18 walker failed to flag an ambiguous-basename citation.  '
        f'``duplicate_basename.py:1`` matches two files in the '
        f'synthetic repo and must contribute to drift (rc=1), not be '
        f'silently skipped; got rc={rc} instead.  This regresses '
        f'AUDIT_V5_3_2 Part 7 P2-1 closure.')


def test_v18_docstring_line_is_trivial(tmp_path, monkeypatch):
    """v5.4 (audit V17/V18 hardening) -- AUDIT_V5_3_2 Part 7 P2-2:
    ``_is_trivial_line`` now flags pure docstring lines (boundary
    ``\\\"\\\"\\\"``, one-line ``\\\"\\\"\\\"foo\\\"\\\"\\\"``, and
    opening-only multiline docstring starts).  A CHANGELOG citing a
    docstring as the "implementation site" must be flagged as drift.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; v5.4 docstring-line test '
            'requires the script module.')

    # Confirm the heuristic directly first.
    assert module._is_trivial_line('"""') is True
    assert module._is_trivial_line("'''") is True
    assert module._is_trivial_line('    """one-liner"""') is True
    assert module._is_trivial_line('"""Module docstring opens here.') is True
    assert module._is_trivial_line("'''multi-line docstring start") is True
    # Sanity: real code lines that happen to contain a triple-quoted
    # substring are NOT misclassified.
    assert module._is_trivial_line('x = """not a docstring"""') is False
    assert module._is_trivial_line('return "foo"') is False

    # End-to-end: synthetic CHANGELOG citing a docstring line.
    fake_pkg = tmp_path / 'lumenairy'
    fake_pkg.mkdir()
    fake_src = fake_pkg / 'docstring_module_for_v18_test.py'
    fake_src.write_text(
        '"""Module docstring start.\n'  # line 1 -- docstring opener
        '\n'                              # line 2 -- blank
        'Some prose.\n'                   # line 3 -- prose body
        '"""\n'                           # line 4 -- docstring closer
        'x = 1\n',                        # line 5 -- real code
        encoding='utf-8',
    )

    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [9.9.9] -- 2099-12-31\n\n'
        '### What changed\n\n'
        '* **Docstring-line citation** at '
        '`lumenairy/docstring_module_for_v18_test.py:4` -- this '
        'bullet cites the docstring closer; V18 must flag it as '
        'drift.\n\n'
        '## [9.9.8] -- 2099-12-30\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )
    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)
    monkeypatch.setattr(module, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(module, '_BASENAME_INDEX_CACHE', None)

    rc = module.audit_block(version='9.9.9', quiet=True)
    assert rc == 1, (
        f'V18 walker failed to flag a docstring-line citation.  Line 4 '
        f'is the docstring-closing ``\\\"\\\"\\\"`` and must register '
        f'as trivial; got rc={rc} instead.  This regresses '
        f'AUDIT_V5_3_2 Part 7 P2-2 closure.')


def test_v18_end_line_verified_for_range_citation(tmp_path, monkeypatch):
    """v5.4 (audit V17/V18 hardening) -- AUDIT_V5_3_2 Part 7 P2-4:
    for ``:START-END`` range citations the walker now verifies BOTH
    endpoints.  Constructs a synthetic citation where START is
    non-trivial but END is trivial (a blank line) and asserts the
    walker returns rc=1.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; v5.4 end-line test '
            'requires the script module.')

    fake_pkg = tmp_path / 'lumenairy'
    fake_pkg.mkdir()
    fake_src = fake_pkg / 'range_module_for_v18_test.py'
    fake_src.write_text(
        'def hello():\n'        # line 1 -- non-trivial
        '    return "world"\n'  # line 2 -- non-trivial
        '\n'                    # line 3 -- BLANK (trivial)
        '\n'                    # line 4 -- BLANK (trivial)
        '\n',                   # line 5 -- BLANK (trivial)
        encoding='utf-8',
    )

    # Citation START=1 (non-trivial) but END=4 (blank).  Pre-hardening
    # walker would have returned 0 (only START checked).  Post-
    # hardening must return 1 because END is trivial.
    synthetic = (
        '# Changelog -- synthetic\n\n'
        '## [9.9.9] -- 2099-12-31\n\n'
        '### What changed\n\n'
        '* **Range-citation with stale END** at '
        '`lumenairy/range_module_for_v18_test.py:1-4` -- START is '
        '``def hello`` (non-trivial) but END=4 is BLANK.  V18 must '
        'flag the END drift.\n\n'
        '## [9.9.8] -- 2099-12-30\n\n'
        '### Stub previous block\n\n'
        '(nothing to verify here)\n'
    )
    fake_changelog = tmp_path / 'CHANGELOG.md'
    fake_changelog.write_text(synthetic, encoding='utf-8')

    monkeypatch.setattr(module, '_CHANGELOG', fake_changelog)
    monkeypatch.setattr(module, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(module, '_BASENAME_INDEX_CACHE', None)

    rc = module.audit_block(version='9.9.9', quiet=True)
    assert rc == 1, (
        f'V18 walker failed to flag a range citation with a trivial '
        f'END line.  START=1 is ``def hello()`` (non-trivial), END=4 '
        f'is BLANK -- pre-v5.4 walker only verified START and would '
        f'pass; got rc={rc} instead.  This regresses AUDIT_V5_3_2 '
        f'Part 7 P2-4 closure.')


def test_v18_bare_return_flagged_as_trivial():
    """v5.4 (audit V17/V18 hardening) -- AUDIT_V5_3_2 Part 7 P2-3:
    fix the dead code at ``_is_trivial_line:124``.  Before v5.4 the
    check was ``stripped.startswith('return\\n')`` which is
    unreachable because ``stripped`` has already had ``\\n`` removed
    by .strip().  Post-fix: a bare ``return`` statement must be
    flagged as trivial.
    """
    module = _load_script_module()
    if module is None:
        pytest.skip(
            'companion script not present; v5.4 bare-return test '
            'requires the script module.')

    # Bare ``return`` on its own line is now trivial.
    assert module._is_trivial_line('    return') is True
    assert module._is_trivial_line('return') is True
    # Returns WITH a value are still non-trivial structural anchors.
    assert module._is_trivial_line('return None') is False
    assert module._is_trivial_line('return foo(bar)') is False
    assert module._is_trivial_line('    return x + 1') is False
