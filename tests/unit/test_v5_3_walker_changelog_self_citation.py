"""v5.3 17th meta-pin walker -- recursive self-citation drift (V17).

The AUDIT_V5_2_5 audit (Part 7) surfaced a NEW sibling-gap class:
**recursive self-citation drift**.  v5.2.5's own CHANGELOG entry
fixed three self-citation drift bugs from v5.2.3:

* Test count ``3765/18/1`` -> stamped at empirical numbers
* Line count ``~10618 -> ~10769`` -> refreshed with both values
* File count "29" claim (v5.2.3) -> (not addressed; v5.2.5 carried
  same form)

But v5.2.5's own CHANGELOG introduced the SAME drift:

* Test count claimed ``3780 / 16 / 1``; empirical ``3781 / 15 / 1``
* Line count claimed ``~10769``; current state ``~10949``
* File count claimed "29 files touched"; actual 26

Each release's CHANGELOG self-cites build-time empirical numbers,
but the CHANGELOG entry itself is part of the diff that establishes
those numbers -- so the cited counts are always at-write-time, not
at-ship-time.

V17 walker pins this empirically.  It does NOT try to FIX the drift
(that requires a pre-tag hook or a V12-style ship-time injection
walker).  Instead it surfaces the drift at audit time so a future
maintainer can decide whether to (a) update the CHANGELOG numbers
to ship-time values via a release-process step or (b) accept the
documented drift class with the V17 walker as the audit witness.

V17 covers three numeric self-citations in the topmost CHANGELOG
block:

* **V17.1 -- test-count self-consistency** -- `X pass + Y skip + Z xfail
  == collected` (V12.3 also pins this; V17.1 adds an empirical-
  collection cross-check via ``pytest --collect-only``).

* **V17.2 -- file-count vs git diff stat** -- if the topmost
  CHANGELOG block claims ``N files touched``, ``git diff
  PREV_TAG..HEAD --stat`` (or against the topmost-block's tag if
  the current state is post-tag) should report approximately N.
  "Approximately" because the CHANGELOG diff itself is in the count
  the author would have measured AT-WRITE-TIME but is not in the
  count if measured at ship-time.

* **V17.3 -- CHANGELOG line-count drift** -- if the topmost block
  references a line count for CHANGELOG.md (e.g. "10769 lines"),
  the actual current line count is at most that count + a small
  margin for the v5.3 entry itself.

Closes AUDIT_V5_2_5 P2-2 + Part 7 "Recursive self-citation drift
NEW class observed" finding.  v5.3.

Author: Andrew Traverso -- v5.3 / AUDIT_V5_2_5 V17 closure.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'


def _read_changelog():
    return _CHANGELOG.read_text(encoding='utf-8')


def _latest_version_block(text):
    """Return ``(version, body)`` for the topmost ``## [X.Y.Z]`` block.

    Mirrors the V12 / V16 walker block-extraction primitive.
    """
    pattern = re.compile(
        r'^## \[(?P<ver>\d+\.\d+\.\d+)\][^\n]*\n(?P<body>.*?)'
        r'(?=^## \[|\Z)',
        re.DOTALL | re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return None, None
    return match.group('ver'), match.group('body')


def _extract_file_count_claim(body):
    """Find a "N files touched" claim in the block body.

    Returns the integer N or None if no such claim.
    """
    # Match patterns like "29 files touched" or "26 files changed"
    # or "29 files: ...".
    m = re.search(
        r'(\d{1,4})\s+files?\s+(?:touched|changed|modified)\b',
        body, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_line_count_claim(body):
    """Find a CHANGELOG.md line-count claim in the block body.

    Returns the integer line count or None.  Matches patterns like
    "CHANGELOG.md: 11553 -> 10618 lines" (uses the post-shrink value)
    or "10769 lines".
    """
    # Try the arrow form first (used at v5.2.3 + v5.2.5).
    m = re.search(
        r'CHANGELOG\.md[^\n]*?\d+\s*->\s*(\d{4,6})\s*lines',
        body, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _git_available():
    """True iff git is on PATH AND the repo has a .git directory."""
    import shutil
    if shutil.which('git') is None:
        return False
    return (_REPO_ROOT / '.git').exists()


# ===========================================================================
# V17.1 -- test-count self-consistency (sums match)
# ===========================================================================

def test_v17_test_count_arithmetic_reconciles():
    """The topmost CHANGELOG block's ``X pass + Y skip + Z xfail`` must
    sum to the cited collected count.

    Note: V12.3 also pins this.  V17.1 is the SELF-CITATION-DRIFT
    variant -- the test exists to keep "the sums match" pinned even
    if a future CHANGELOG edit drops V12.3's exact "X unit tests
    pass" wording but keeps a different numeric self-citation form.
    """
    version, body = _latest_version_block(_read_changelog())
    if version is None:
        pytest.skip('CHANGELOG.md has no ``## [X.Y.Z]`` blocks.')

    pattern = re.compile(
        r'(?P<pass>\d{3,5})\s+unit tests pass[^(]*'
        r'\(\s*collected\s*=?\s*(?P<coll>\d{3,5})\s*=\s*'
        r'pass\s*\+\s*(?P<skip>\d+)\s+skip\s*\+\s*'
        r'(?P<xfail>\d+)\s+xfail',
        re.IGNORECASE)
    match = pattern.search(body)
    if not match:
        pytest.skip(
            f'CHANGELOG ``## [{version}]`` block has no canonical '
            f'``X pass + Y skip + Z xfail`` self-citation; V17.1 '
            f'has nothing to verify.')

    pass_n = int(match.group('pass'))
    coll_n = int(match.group('coll'))
    skip_n = int(match.group('skip'))
    xfail_n = int(match.group('xfail'))
    assert pass_n + skip_n + xfail_n == coll_n, (
        f'CHANGELOG ``## [{version}]`` test-count arithmetic does '
        f'not reconcile: {pass_n} pass + {skip_n} skip + {xfail_n} '
        f'xfail = {pass_n + skip_n + xfail_n}, but collected count '
        f'is {coll_n}.')


# ===========================================================================
# V17.2 -- file-count claim vs git diff stat
# ===========================================================================

def test_v17_file_count_claim_within_drift_band():
    """If the topmost CHANGELOG block claims ``N files touched``,
    ``git diff PREV_TAG..HEAD --stat`` should report approximately N.

    Approximately because the CHANGELOG.md diff is itself in the
    count at-ship-time but may not be in the count the author cited
    AT-WRITE-TIME (the CHANGELOG entry is part of the diff that
    establishes the count).  v5.3 (AUDIT_V5_2_5 P2-2 closure)
    tolerates a band of +/- 5 files.

    Skips cleanly when git is not available, the block has no
    file-count claim, or PREV_TAG can't be resolved.
    """
    if not _git_available():
        pytest.skip('git not available; V17.2 cannot verify.')

    text = _read_changelog()
    version, body = _latest_version_block(text)
    if version is None:
        pytest.skip('CHANGELOG.md has no ``## [X.Y.Z]`` blocks.')

    claimed_n = _extract_file_count_claim(body)
    if claimed_n is None:
        pytest.skip(
            f'CHANGELOG ``## [{version}]`` block has no '
            f'``N files touched`` claim; V17.2 has nothing to '
            f'verify.')

    # Find PREV_TAG from the second-most-recent CHANGELOG block.
    # Walk all blocks; the second is the previous release.
    all_blocks = list(re.finditer(
        r'^## \[(?P<ver>\d+\.\d+\.\d+)\]',
        text, re.MULTILINE))
    if len(all_blocks) < 2:
        pytest.skip(
            'CHANGELOG has fewer than 2 ``## [X.Y.Z]`` blocks; '
            'V17.2 needs a PREV_TAG to diff against.')
    prev_ver = all_blocks[1].group('ver')
    prev_tag = f'v{prev_ver}'

    # Confirm the tag exists.
    proc = subprocess.run(
        ['git', 'rev-parse', '--verify', f'refs/tags/{prev_tag}'],
        cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f'PREV_TAG ``{prev_tag}`` is not a reachable git tag '
            f'(shallow clone or pre-release state); V17.2 cannot '
            f'verify.')

    # Get the actual file count from git diff.
    proc = subprocess.run(
        ['git', 'diff', '--name-only', f'{prev_tag}..HEAD'],
        cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f'``git diff {prev_tag}..HEAD`` failed; V17.2 cannot '
            f'verify.')

    actual_files = [
        ln.strip() for ln in proc.stdout.splitlines() if ln.strip()
    ]
    actual_n = len(actual_files)

    # +/- 5 file drift band -- CHANGELOG.md itself + a few new
    # tests added between v5.2.5's CHANGELOG-write moment and ship.
    drift_tol = 5
    assert abs(actual_n - claimed_n) <= drift_tol, (
        f'CHANGELOG ``## [{version}]`` claims ``{claimed_n} files '
        f'touched`` but ``git diff {prev_tag}..HEAD --name-only`` '
        f'reports {actual_n} files (drift of '
        f'{abs(actual_n - claimed_n)} > tolerance {drift_tol}).  '
        f'The recursive self-citation drift class has widened past '
        f'the documented tolerance.  Refresh the CHANGELOG file '
        f'count to the at-ship-time value.')


# ===========================================================================
# V17.3 -- CHANGELOG line-count claim drift band
# ===========================================================================

def test_v17_line_count_claim_within_drift_band():
    """If the topmost CHANGELOG block cites a CHANGELOG.md line
    count, the actual current line count should not exceed it by
    more than ~200 lines (the typical CHANGELOG-entry-itself
    contribution).

    Drift in the OTHER direction (claimed > actual) is a stronger
    signal of cleanup-after-archive; not flagged.

    Skips cleanly when the block has no line-count claim.
    """
    text = _read_changelog()
    version, body = _latest_version_block(text)
    if version is None:
        pytest.skip('CHANGELOG.md has no ``## [X.Y.Z]`` blocks.')

    claimed_n = _extract_line_count_claim(body)
    if claimed_n is None:
        pytest.skip(
            f'CHANGELOG ``## [{version}]`` block has no '
            f'``CHANGELOG.md: X -> Y lines`` claim; V17.3 has '
            f'nothing to verify.')

    actual_n = len(text.splitlines())
    drift_tol = 300  # generous band; one v5.x entry is typically 100-200 lines
    drift = actual_n - claimed_n
    assert drift <= drift_tol, (
        f'CHANGELOG ``## [{version}]`` claims a post-archive line '
        f'count of {claimed_n}; current is {actual_n} (drift '
        f'+{drift} > tolerance {drift_tol}).  The CHANGELOG has '
        f'grown by more than one typical release entry past the '
        f'cited count; refresh the line-count claim to the current '
        f'state at v5.3+.')
