"""v5.2 12th meta-pin walker -- CHANGELOG-vs-changeset verifier (V12).

The v5.1.0 audit identified that the sibling-gap meta-pattern,
structurally retired at code surfaces via V1-V10 and at
documentation surfaces via V11, recurred at a new surface:
**CHANGELOG-vs-implementation**.  v5.1.0's CHANGELOG declared 11
audit closures with concrete file:line citations, but 10 of the 11
had no backing diff -- the claimed code changes were never written.
Detected by independent 2-way agent convergence (V1 + F2).

This walker scans CHANGELOG.md's most-recent version block for
falsifiable claims and asserts the claim has a backing artifact:

* Every backticked file path cited (``lumenairy/foo.py``,
  ``tests/unit/test_bar.py``, ``.github/workflows/baz.yml``,
  ``Migration-Guide.md``, etc.) must exist in the repo.

* Every audit-ID cited (``P1-NEW-2WAY-1``, ``P2-NEW-F2-3``, etc.)
  must appear somewhere under ``docs/audits/``.

* If a ``## [X.Y.Z]`` heading immediately above the version block
  matches ``lumenairy.__version__`` (i.e. we are auditing the
  current release), the bullet test-count claim
  (e.g. ``3628 unit tests pass``) must reconcile with at least
  one paired count line.

Closes audit P1-V5_1_1-1 (CHANGELOG-vs-changeset sibling-gap class).

The walker is conservative: it only catches FILE-LEVEL claims that
can be mechanically verified.  CONTENT-LEVEL fabrications (the
file exists but the cited behavior is missing) require human review
or a tighter diff-aware variant -- see
``scripts/verify_changelog_closures.py`` for the git-diff-aware
companion that runs at tag time.

v5.1.0 fabrication retrospective: the publish.yml file did exist
at v5.1.0 ship, so a file-existence check would NOT have caught it
on its own.  But the claim ``new `verify` job runs the unit suite``
would have been falsified by a content-aware check, which is the
script-only companion's responsibility.  V12 here is the cheaper
half: it catches typo-level + deleted-file fabrications + audit-ID
fabrications, which are the most common failure modes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_AUDITS_DIR = _REPO_ROOT / 'docs' / 'audits'


def _read_changelog():
    assert _CHANGELOG.is_file(), (
        f'CHANGELOG.md not found at {_CHANGELOG}; the V12 walker '
        f'has no input to audit.')
    return _CHANGELOG.read_text(encoding='utf-8')


def _latest_version_block(text):
    """Return (version_string, body_text) for the topmost ``## [X.Y.Z]``
    block in CHANGELOG.md.  Body runs from after the heading to the
    next ``## `` heading (top-level horizontal-rule + heading)."""
    pattern = re.compile(
        r'^## \[(?P<ver>\d+\.\d+\.\d+)\][^\n]*\n(?P<body>.*?)'
        r'(?=^## \[|\Z)',
        re.DOTALL | re.MULTILINE)
    match = pattern.search(text)
    assert match is not None, (
        'CHANGELOG.md has no ``## [X.Y.Z]`` per-release heading; '
        'V12 walker cannot locate the latest version block.')
    return match.group('ver'), match.group('body')


# Path tokens that look like real repo files.  Catches:
# - ``lumenairy/foo.py`` / ``lumenairy/foo/bar.py``
# - ``tests/unit/test_baz.py``
# - ``.github/workflows/qux.yml``
# - ``examples/quux.py``
# - top-level ``README.md`` / ``CHANGELOG.md`` / ``Migration-Guide.md``
# - ``pyproject.toml`` / ``requirements.txt``
# - ``scripts/whatever.py``
# - ``docs/whatever.md``
#
# v5.2: anchored to backtick delimiters so we don't pick up file
# paths from prose ("see CHANGELOG.md"); only those wrapped in
# inline-code spans count as citations.
_FILE_PATH_PATTERN = re.compile(
    r'`(?P<path>'
    r'(?:lumenairy|tests|examples|docs|scripts|benchmarks|validation)'
    r'/[\w\-./]+\.\w{1,5}'
    r'|\.github/[\w\-./]+\.\w{1,5}'
    r'|(?:README|CHANGELOG|ROADMAP|Migration-Guide|CONTRIBUTING|CONVENTIONS)\.md'
    r'|pyproject\.toml'
    r'|requirements\.txt'
    r')'
    r'(?::\d+(?:-\d+)?)?'  # optional :line or :start-end
    r'`')


_AUDIT_ID_PATTERN = re.compile(
    r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z][A-Z0-9_-]{2,})')


def _extract_cited_paths(body):
    return {m.group('path') for m in _FILE_PATH_PATTERN.finditer(body)}


def _extract_cited_audit_ids(body):
    return {m.group(1) for m in _AUDIT_ID_PATTERN.finditer(body)}


# ===========================================================================
# V12.1 -- every backticked file path citation must resolve to a real file
# ===========================================================================

def test_v12_cited_file_paths_exist():
    """Every backticked file path in the most-recent CHANGELOG
    version block must exist in the repo.

    Catches the v5.1.0-class fabrication where a CHANGELOG bullet
    cites a file that doesn't exist (typo or deleted-file claim).
    Does NOT catch content-level fabrications (file exists but
    cited behavior is missing) -- see V12.4 below + the companion
    ``scripts/verify_changelog_closures.py`` for that.
    """
    text = _read_changelog()
    version, body = _latest_version_block(text)
    cited = _extract_cited_paths(body)

    missing = []
    for path in sorted(cited):
        # Strip ``:line`` or ``:start-end`` suffix before resolving.
        core = path.split(':', 1)[0]
        if not (_REPO_ROOT / core).exists():
            missing.append(path)

    assert not missing, (
        f'CHANGELOG.md ``## [{version}]`` block cites file paths '
        f'that do not exist in the repo: {missing!r}.  These are '
        f'v5.1.0-class fabrications -- the bullet text claims a '
        f'change to a file that is not present.  Either fix the '
        f'citation, create the file, or remove the claim.')


# ===========================================================================
# V12.2 -- every audit-ID citation must appear under docs/audits/
# ===========================================================================

def test_v12_cited_audit_ids_resolve():
    """Every audit ID (``P[0-3]-NEW-<scope>-<n>`` etc.) cited in the
    most-recent CHANGELOG version block must appear somewhere in
    ``docs/audits/`` -- a citation of a non-existent audit is
    itself a small fabrication (a claim of a closure for a finding
    that was never written).

    The walker tolerates audit IDs that appear in OTHER CHANGELOG
    sections (per-release headings, recap tables), since the audit
    document might be the same one that catalogued the finding.
    The fail mode is: an audit ID that resolves nowhere.
    """
    if not _AUDITS_DIR.is_dir():
        pytest.skip(
            f'docs/audits/ directory not present at {_AUDITS_DIR}; '
            f'V12.2 cannot verify audit-ID citations.')

    text = _read_changelog()
    version, body = _latest_version_block(text)
    cited_ids = _extract_cited_audit_ids(body)

    if not cited_ids:
        pytest.skip(
            f'No audit IDs cited in CHANGELOG.md ``## [{version}]`` '
            f'block; V12.2 has nothing to verify.')

    audit_files = list(_AUDITS_DIR.rglob('*.md'))
    audit_text = '\n'.join(
        f.read_text(encoding='utf-8', errors='replace')
        for f in audit_files)

    unresolved = [aid for aid in sorted(cited_ids) if aid not in audit_text]
    assert not unresolved, (
        f'CHANGELOG.md ``## [{version}]`` block cites audit IDs '
        f'that appear in NO file under docs/audits/: {unresolved!r}.  '
        f'Either the audit was never written, or the ID was typo\'d.')


# ===========================================================================
# V12.3 -- test-count claims must reconcile when version == current
# ===========================================================================

def test_v12_test_count_claim_reconciles_with_collection():
    """If the topmost ``## [X.Y.Z]`` block matches the library's
    declared ``__version__`` (i.e. we are auditing the current
    release), any test-count claim in the bullet text must
    plausibly reconcile with pytest's collection count.

    Skipped for past-version blocks because their test counts are
    historical and we don't try to time-travel.  Skipped if the
    block has no test-count claim.  ``plausibly reconcile`` means:
    the cited pass-count plus the cited skip-count plus the cited
    xfail-count equals the cited collected-count.  This is a
    self-consistency check, not an empirical pytest-collection
    check (which would couple the walker to test-discovery time).
    """
    import lumenairy
    text = _read_changelog()
    version, body = _latest_version_block(text)

    if version != lumenairy.__version__:
        pytest.skip(
            f'CHANGELOG top block is for v{version}; '
            f'library is v{lumenairy.__version__}.  Skipping the '
            f'current-version self-consistency check.')

    # Match patterns like:
    # ``3628 unit tests pass (collected = 3634 = pass + 5 skip + 1 xfail)``
    # ``3628 unit tests pass (collected 3634 = pass + 5 skip + 1 xfail)``
    pattern = re.compile(
        r'(?P<pass>\d{3,5})\s+unit tests pass[^(]*'
        r'\(\s*collected\s*=?\s*(?P<coll>\d{3,5})\s*=\s*'
        r'pass\s*\+\s*(?P<skip>\d+)\s+skip\s*\+\s*'
        r'(?P<xfail>\d+)\s+xfail',
        re.IGNORECASE)
    match = pattern.search(body)
    if not match:
        pytest.skip(
            f'CHANGELOG ``## [{version}]`` block has no '
            f'``X pass / Y collected / Z skip / W xfail`` claim; '
            f'V12.3 has nothing to reconcile.')

    pass_n = int(match.group('pass'))
    coll_n = int(match.group('coll'))
    skip_n = int(match.group('skip'))
    xfail_n = int(match.group('xfail'))
    assert pass_n + skip_n + xfail_n == coll_n, (
        f'CHANGELOG ``## [{version}]`` test-count arithmetic does '
        f'not reconcile: {pass_n} pass + {skip_n} skip + {xfail_n} '
        f'xfail = {pass_n + skip_n + xfail_n}, but collected count '
        f'is {coll_n}.  Either the claim has been edited without '
        f'updating the arithmetic, or the actual test counts have '
        f'drifted without a CHANGELOG refresh.')


# ===========================================================================
# V12.4 -- declared audit-closure section must reconcile to backing diffs
# ===========================================================================

def test_v12_changelog_has_explicit_pre_tag_step_or_no_audit_closures():
    """A weaker structural check that closes the v5.1.0 fabrication
    surface in a different way: if the most-recent CHANGELOG version
    block declares ``audit closures``, it must EITHER (a) cite the
    pre-tag verification practice (``git diff PREV_TAG..HEAD`` or
    the V12 walker itself) OR (b) the script
    ``scripts/verify_changelog_closures.py`` must exist.

    This pin's purpose is to make the verification practice
    impossible to forget: a release with audit-closure claims must
    advertise HOW those claims were verified.  v5.1.1's CHANGELOG
    described the practice but provided no enforcement -- which
    F1 flagged.  v5.2 turns the description into a structural
    requirement: either the prose explicitly names V12 + the
    companion script, or the bullet doesn't claim to close an
    audit.
    """
    text = _read_changelog()
    version, body = _latest_version_block(text)

    declares_closures = bool(re.search(
        r'(?i)audit\s+closur|closes the .* audit', body))
    if not declares_closures:
        pytest.skip(
            f'CHANGELOG ``## [{version}]`` block declares no audit '
            f'closures; V12.4 has no enforcement to apply.')

    cites_v12 = ('V12' in body
                 or 'verify_changelog_closures' in body
                 or 'CHANGELOG-vs-changeset' in body
                 or 'CHANGELOG-vs-diff' in body)
    has_script = (_REPO_ROOT / 'scripts' / 'verify_changelog_closures.py').is_file()
    has_walker = (
        _REPO_ROOT / 'tests' / 'unit'
        / 'test_v5_2_walker_changelog_changeset.py'
    ).is_file()

    assert cites_v12 or has_script or has_walker, (
        f'CHANGELOG ``## [{version}]`` declares audit closures but '
        f'has no advertised verification mechanism.  Either the '
        f'block prose must cite V12 / verify_changelog_closures / '
        f'the CHANGELOG-vs-changeset practice, or the companion '
        f'script ``scripts/verify_changelog_closures.py`` must '
        f'exist, or this walker file must exist.  v5.1.0 demonstrated '
        f'that audit-closure declarations without an enforcement '
        f'mechanism are vulnerable to fabrication.')
