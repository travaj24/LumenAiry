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
        stdin=subprocess.DEVNULL,
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


# ===========================================================================
# V18.5 -- the CONTENT check V18 deliberately does not make (V-D2, 2026-09-19)
# ===========================================================================
#
# V18 asks "is the cited line non-trivial?".  It does NOT ask "is it the RIGHT
# line", and its own docstring says so ("NECESSARY-BUT-NOT-SUFFICIENT").  That
# gap is not theoretical.  MEASURED 2026-09-19 on ``refactor/wave5-hygiene-2``
# (VERIFY-WAVE5-HYGIENE2 V-D2): V18 read ``ok=107 drift=0 total=107``, rc=0 on
# both builds, while SIXTEEN citations in the 5.47.0 block named the wrong
# line -- fifteen never re-anchored after H2-1/H2-2/H2-3 moved lines in
# ``mft.py`` and ``carrier.py``, and one (``mft.py:611-620``) re-anchored BY
# HAND with a ``+34`` shift where its content had moved ``+38``, so the
# sentence's own reading sat two lines outside its cited range.  Every one of
# the sixteen landed on some other real line, which is exactly what V18 passes.
#
# ``scripts/reanchor_citations.py`` makes the content statement: it takes the
# line a citation named at a BASE commit, finds that line's CONTENT in the tree
# as it is now, and reports any citation that does not name it.  These three
# ids run it over the 5.47.0 block, assert the tool covers the files that block
# cites, and prove on a synthetic pair that the content check sees a drift V18
# passes.

_REANCHOR = _REPO_ROOT / 'scripts' / 'reanchor_citations.py'
#: The commit the 5.47.0 block's citations were written against -- the base of
#: the Wave 5 hygiene-2 work.  Not a moving target: a block's citations are
#: anchored once, against the tree whose line numbers they quoted.
_V547_BASE = 'f4f18851'
_V547_BLOCK = '[5.47.0]'


def _load_reanchor_module(name):
    """Import ``scripts/reanchor_citations.py`` by path, like V18's own."""
    spec = importlib.util.spec_from_file_location(name, _REANCHOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_changelog_or_fail(module):
    """``git show <base>:CHANGELOG.md``, with the two failure modes separated.

    A gate that cannot reach its base commit has NOT verified anything, so it
    fails rather than skips (TESTING_STANDARDS S3 -- "never ``pytest.skip`` on
    a resource check"; a skipped citation gate is how V-D2 shipped).  But the
    reader is owed the difference between "a citation is wrong" and "git
    cannot see this repository from here", so the two say different things.

    MEASURED 2026-09-19: running this suite from WSL against a WINDOWS git
    worktree, ``git rev-parse --git-dir`` itself fails -- the worktree's
    ``.git`` file holds ``gitdir: D:/.../.git/worktrees/<name>``, a path that
    does not exist under WSL.  The same condition already makes
    ``test_v16_synthetic_fabrication_is_caught`` red on that lane, on the base
    tree as well as on this one, so it is an environment fact and not a
    finding.  On a CI clone and on Windows, both resolve.
    """
    probe = subprocess.run(['git', 'rev-parse', '--git-dir'], cwd=_REPO_ROOT,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8')
    if probe.returncode != 0 or 'not a git repository' in (probe.stderr or ''):
        pytest.fail(
            f"ENVIRONMENT, not a citation finding: git cannot resolve this "
            f"repository from {_REPO_ROOT} ({(probe.stderr or '').strip()[:160]}).  "
            f"This is the WSL-against-a-Windows-worktree condition that also "
            f"makes test_v16_synthetic_fabrication_is_caught red on that lane, "
            f"on the base tree too.  The citation content gate cannot run "
            f"without the base commit, and it does not skip.")
    before = subprocess.run(
        ['git', 'show', f'{_V547_BASE}:CHANGELOG.md'],
        cwd=_REPO_ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8')
    if before.returncode != 0 or not before.stdout:
        pytest.fail(
            f"git show {_V547_BASE}:CHANGELOG.md failed (rc="
            f"{before.returncode}).  The content check needs the base tree; a "
            f"clone that cannot reach it must FAIL here rather than skip -- a "
            f"skipped citation gate is how V-D2 shipped.")
    return before.stdout


def test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files():
    """The content checker is a repo tool, and its coverage is the whole point.

    It lived in ``validation/probe_wp_b11c/`` where nothing ran it, and its
    ``OWNED`` map named four lens / rcwa files, so on a branch that moved
    ``mft.py`` and ``carrier.py`` it printed ``0 re-anchored`` -- a green that
    meant nothing.  Its home and its coverage are both part of the fix, so
    both are asserted.
    """
    assert _REANCHOR.is_file(), (
        f'{_REANCHOR.name} is missing from scripts/.  V18 alone cannot tell a '
        f'right citation from a wrong one that lands on a non-trivial line; '
        f'this tool is the half that can.')
    src = _REANCHOR.read_text(encoding='utf-8')
    ast.parse(src)
    module = _load_reanchor_module('_reanchor_v18_5a')
    # Every file the 5.47.0 block cites by a source line must be OWNED, or the
    # check cannot see it.  Derived from the block itself, not listed here.
    base_text = _base_changelog_or_fail(module)
    lo, hi = module._block_span(base_text, _V547_BLOCK)
    cited = {t.rsplit(':', 1)[0]
             for t in module.CITE_RE.findall(base_text[lo:hi])
             if not t.startswith('`')}
    # Only tails that name a real shipped module: a citation into a test file,
    # a script or a doc is outside this tool's remit.  The tail is kept whole
    # -- ``rcwa/_core.py`` and ``pmm/oned.py`` are only resolvable WITH their
    # path fragment, and that fragment is what the CHANGELOG already writes.
    uncovered = sorted(
        t for t in cited
        if module.owner_of(t) is None
        and any((_REPO_ROOT / 'lumenairy').rglob(t.rsplit('/', 1)[-1])))
    assert not uncovered, (
        f"the 5.47.0 block cites {uncovered} by source line, and "
        f"reanchor_citations.py's OWNED map does not cover them, so those "
        f"citations cannot be content-checked at all.  That blindness is the "
        f"mechanical root cause of V-D2 -- the tool reported '0 re-anchored' "
        f"on a branch with fifteen stale citations because none of the files "
        f"it had moved were in the map.")


def test_v18_5_the_5_47_0_block_citations_name_the_right_lines():
    """Every owned ``path.py:N`` in the 5.47.0 block names the CONTENT its
    base commit's citation named.

    Two-sided by construction: the tool reports a citation that is stale
    (``changed``) AND one whose anchor it cannot find at all (``notes``), and
    this asserts both are empty.  Nothing here pins a line NUMBER -- every
    number is derived from ``git show`` of the base commit and from the
    working tree, so ordinary evolution of either file moves the expectation
    with it.

    PREMISE first, so a tool that silently stopped looking cannot pass: the
    block must still contain owned citations to check.
    """
    if not _REANCHOR.is_file():
        pytest.fail('scripts/reanchor_citations.py is missing; see '
                    'test_v18_5_companion_reanchor_tool_exists_and_covers'
                    '_the_cited_files')
    module = _load_reanchor_module('_reanchor_v18_5b')
    base_text = _base_changelog_or_fail(module)
    lo, hi = module._block_span(base_text, _V547_BLOCK)
    owned_cites = [t for t in module.CITE_RE.findall(base_text[lo:hi])
                   if not t.startswith('`')
                   and module.owner_of(t.rsplit(':', 1)[0])]
    assert len(owned_cites) >= 10, (
        f'PREMISE FAILED: only {len(owned_cites)} owned source-line citations '
        f'found in the {_V547_BLOCK} block at {_V547_BASE}.  MEASURED '
        f'2026-09-19: 19.  A collapse toward zero means the tool stopped '
        f'recognising the citation spelling, and the assertion below would be '
        f'vacuously green.')

    _txt, changed, notes = module.reanchor(base=_V547_BASE, block=_V547_BLOCK)
    assert not changed and not notes, (
        f'{len(changed)} citation(s) in the {_V547_BLOCK} CHANGELOG block do '
        f'not name the line whose CONTENT they named at {_V547_BASE}, and '
        f'{len(notes)} could not be anchored at all.  V18 passes all of these '
        f'-- every one lands on some other real line -- which is the blind '
        f'spot this gate exists for.\n'
        + '\n'.join('  ' + c for c in changed + notes)
        + f'\n\nFix: python scripts/reanchor_citations.py --base {_V547_BASE} '
          f'--block "{_V547_BLOCK}"   (content-based and idempotent; '
          f'--check writes nothing).')


def test_v18_5_the_content_check_sees_a_drift_v18_passes():
    """Falsification arm: a citation drifted onto a DIFFERENT non-trivial line.

    Without this, the id above could be green because the checker had quietly
    stopped checking.  A synthetic base / current pair is built in memory --
    one inserted comment line above a definition -- and two things are
    asserted: V18's own triviality rule accepts the drifted-to line (so this
    really is inside V18's blind spot), and the content locator follows the
    definition to its new line.
    """
    if not _REANCHOR.is_file():
        pytest.fail('scripts/reanchor_citations.py is missing')
    module = _load_reanchor_module('_reanchor_v18_5c')
    v18 = _load_script_module()
    base_src = ['def alpha():', '    return 1', '', 'def beta():',
                '    return 2']
    now_src = ['# a comment the refactor inserted above alpha'] + base_src
    # The citation named ``def alpha():`` at base line 1.  After the insertion
    # line 1 holds a comment, which V18 reads as a perfectly good anchor.
    if v18 is not None:
        assert v18._is_trivial_line(now_src[0]) is False, (
            "the synthetic drift target must be a line V18 ACCEPTS, otherwise "
            "this arm proves nothing about V18's blind spot.")
    ctx = [None, None, base_src[0], base_src[1], base_src[2]]
    assert module.locate(base_src[0], ctx, now_src, prefer=1) == (2, 'unique'), (
        f'the content locator did not follow the definition it was given: got '
        f'{module.locate(base_src[0], ctx, now_src, prefer=1)!r}, expected '
        f'(2, "unique").  A locator that returned the cited number unchanged '
        f'would make the 5.47.0 gate vacuous.')
    # And the unchanged direction: content still at its cited line is NOT a
    # drift, however many twins it has elsewhere in the file (the
    # ``rcwa/_core.py:1578`` / ``:1667`` pair that used to read "ambiguous").
    twin = base_src + base_src
    assert module.locate(base_src[0], ctx, twin, prefer=1) == (1, 'unchanged')
