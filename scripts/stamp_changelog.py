#!/usr/bin/env python3
# v5.3.2 (ROADMAP V17 fix companion -- pre-tag ship-time-stamp injection):
# v5.4.1 (AUDIT_V5_4_0 P2 #3): extended with Net LOC: +X / -Y stamp to close
# the 619-line tombstone-deletion drift class flagged in the v5.4.0 audit.
"""stamp_changelog.py -- CHANGELOG ship-time-stamp injection script.

Companion to the V17 walker
(``tests/unit/test_v5_3_walker_changelog_self_citation.py``).  The V17
walker DETECTS recursive self-citation drift: each release's CHANGELOG
self-cites build-time empirical numbers (test counts, file counts,
line counts), but the CHANGELOG entry itself is part of the diff that
establishes those numbers -- so the cited counts are always at-write-
time, not at-ship-time.  This script FIXES the drift by stamping the
CHANGELOG block with empirical values just before tag commit.

The script is dry-run by default -- it prints a unified diff of the
proposed rewrite and exits without touching CHANGELOG.md.  Passing
``--apply`` triggers the actual write.  This prevents accidental
rewrites and forces the maintainer to inspect the proposed change
before committing.

Patterns rewritten in the topmost ``## [X.Y.Z]`` block:

* ``X unit tests pass (collected = Y = pass + Z skip + W xfail)``
  -- update X / Y / Z / W to current empirical values.
* ``N files touched`` / ``N files changed`` / ``N files modified``
  -- update N to ``git diff PREV_TAG..HEAD --name-only | wc -l``.
* ``CHANGELOG.md: OLD -> NEW lines`` -- update NEW to current line count.
* ``Net LOC: +X / -Y`` (optionally followed by ``vs vA.B.C``)
  -- update X (insertions) and Y (deletions) to
  ``git diff PREV_TAG..HEAD --shortstat``.  Added v5.4.1 to close the
  619-line tombstone-deletion drift class observed in v5.4.0.

If a pattern is NOT found in the block, the script logs "skipped: no
pattern to update" and proceeds.  It does NOT fabricate new pattern
sites.

Usage:
    python scripts/stamp_changelog.py
        -- dry-run against the topmost block, --quick mode (default)

    python scripts/stamp_changelog.py --apply
        -- actually rewrite CHANGELOG.md with current empirical values

    python scripts/stamp_changelog.py --full
        -- run the full unit suite (slow, ~5 min) to get pass/skip/xfail
           counts instead of collect-only

    python scripts/stamp_changelog.py --version 5.2.3
        -- back-stamp a past version block instead of the topmost

Exit codes:
    0  -- clean (dry-run with no changes needed, OR --apply succeeded)
    1  -- drift detected but --apply not set (user can choose)
    2  -- skip-clean (no patterns to verify; nothing to do)
    3  -- input error (CHANGELOG not parseable, etc.)

Workflow at tag time (see docs/release-process.md):
    1. Write CHANGELOG.md entry with at-write-time placeholder counts.
    2. ``git add CHANGELOG.md`` along with all other release files.
    3. ``python scripts/stamp_changelog.py --quick --apply``
    4. ``git add CHANGELOG.md`` again to include the stamp updates.
    5. ``git commit``, ``git tag``, ``git push``.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_TESTS_UNIT = _REPO_ROOT / 'tests' / 'unit'


# Block-extraction regex -- copied verbatim from V17 walker's
# ``_latest_version_block`` primitive (which itself mirrors V12).
_VERSION_BLOCK_PATTERN = re.compile(
    r'^## \[(?P<ver>\d+\.\d+\.\d+)\][^\n]*\n(?P<body>.*?)'
    r'(?=^## \[|\Z)',
    re.DOTALL | re.MULTILINE)


# Test-count self-citation pattern.  Matches the canonical form used
# across v5.x CHANGELOG entries:
#     "3848 unit tests pass (collected = 3866 = pass + 17 skip + 1 xfail)"
# Captures groups: pass, coll, skip, xfail.  The captured spans are
# substituted in place; surrounding text is left untouched.
_TEST_COUNT_PATTERN = re.compile(
    r'(?P<pass>\d{3,5})(?P<m1>\s+unit tests pass[^(]*'
    r'\(\s*collected\s*=?\s*)(?P<coll>\d{3,5})(?P<m2>\s*=\s*'
    r'pass\s*\+\s*)(?P<skip>\d+)(?P<m3>\s+skip\s*\+\s*)'
    r'(?P<xfail>\d+)(?P<m4>\s+xfail)',
    re.IGNORECASE)


# File-count self-citation pattern.  Matches:
#     "29 files touched"
#     "26 files changed"
#     "29 files modified"
# Number captured for replacement.
_FILE_COUNT_PATTERN = re.compile(
    r'(?P<n>\d{1,4})(?P<m>\s+files?\s+(?:touched|changed|modified)\b)',
    re.IGNORECASE)


# CHANGELOG line-count self-citation pattern.  Matches:
#     "CHANGELOG.md: 10618 -> 10769 lines"
#     "CHANGELOG.md reduced from 10949 to 5236 lines"
# The post-arrow / post-"to" number is the value the author last
# observed; that is the number to refresh.
_LINE_COUNT_ARROW_PATTERN = re.compile(
    r'(CHANGELOG\.md[^\n]*?\d+\s*->\s*)(?P<n>\d{4,6})(\s*lines)',
    re.IGNORECASE)
_LINE_COUNT_FROM_TO_PATTERN = re.compile(
    r'(CHANGELOG\.md[^\n]*?\bfrom\s+\d+\s+to\s+)(?P<n>\d{4,6})(\s*lines)',
    re.IGNORECASE)


# Net-LOC self-citation pattern (v5.4.1 addition).  Matches the
# canonical v5.x phrasing:
#     "Net LOC: +13759 / -452 vs v5.3.2."
#     "Net LOC: +2766 / -11."
#     "Net LOC: +1500/-200 vs v1.2.3"   (no spaces around slash, allowed)
# Captures:
#   ins   -- insertions count (the "+" number)
#   del_  -- deletions count (the "-" number; trailing underscore avoids
#            shadowing the python builtin ``del``)
# The trailing ``vs vX.Y.Z`` is optional; both forms are stamped.
# Group structure (5 groups total):
#   1 = "Net LOC: +" prefix (case-insensitive matching, but case
#       preserved by Python regex's capture, so the original prefix
#       stays intact on rewrite).
#   2 = ins (named ``ins``)
#   3 = " / -" middle glue (with optional spaces)
#   4 = del_ (named ``del_``)
#   5 = " vs vX.Y.Z" suffix OR a sentence-terminator (period / end-of-line)
#
# The suffix capture is the tricky part:
#   - It must consume "vs vX.Y.Z" or "vs X.Y.Z" so the trailing
#     version label round-trips intact on rewrite.
#   - It must NOT consume the sentence-terminating period that
#     follows "vs v5.3.2." in the canonical v5.4.0 phrasing --
#     hence the dotted-version atom ``\d+(?:\.\d+)*`` rather than
#     a loose ``[0-9.]+`` character class.
#   - When no ``vs`` clause is present, the suffix is a zero-width
#     lookahead at a whitespace / punctuation boundary so the
#     numbers can be replaced without disturbing surrounding text.
_NET_LOC_PATTERN = re.compile(
    r'(?P<pre>Net\s+LOC\s*:\s*\+)'
    r'(?P<ins>\d+)'
    r'(?P<mid>\s*/\s*-)'
    r'(?P<del_>\d+)'
    r'(?P<suf>\s+vs\s+v?\d+(?:\.\d+)*|(?=[\s.,)]|$))',
    re.IGNORECASE)


# ---------------------------------------------------------------------------
# CHANGELOG parsing -- borrowed from V17 walker
# ---------------------------------------------------------------------------

def _read_changelog():
    """Return the CHANGELOG.md text or raise SystemExit(3)."""
    if not _CHANGELOG.is_file():
        print(f'ERROR: CHANGELOG.md not found at {_CHANGELOG}; nothing '
              f'to stamp.  (exit 3)', file=sys.stderr)
        raise SystemExit(3)
    return _CHANGELOG.read_text(encoding='utf-8')


def _latest_version_block(text):
    """Return ``(version, body, start, end)`` for the topmost
    ``## [X.Y.Z]`` block in CHANGELOG.md.

    ``start`` is the body's start offset, ``end`` its end offset --
    both relative to ``text``.  ``body`` runs from the line AFTER the
    heading to the next ``## [`` heading (or EOF).

    Returns ``(None, None, None, None)`` if no block found.
    """
    match = _VERSION_BLOCK_PATTERN.search(text)
    if match is None:
        return None, None, None, None
    return (match.group('ver'), match.group('body'),
            match.start('body'), match.end('body'))


def _find_version_block(text, version):
    """Return ``(version, body, start, end)`` for a specific version.

    If ``version`` is None, returns the topmost block.
    Returns ``(None, None, None, None)`` if not found.
    """
    if version is None:
        return _latest_version_block(text)
    for match in _VERSION_BLOCK_PATTERN.finditer(text):
        if match.group('ver') == version:
            return (match.group('ver'), match.group('body'),
                    match.start('body'), match.end('body'))
    return None, None, None, None


# ---------------------------------------------------------------------------
# Empirical-count helpers
# ---------------------------------------------------------------------------

def _run(cmd, timeout=600):
    """Run a subprocess from the repo root.  Returns ``(rc, stdout)``.
    Captures stderr to stdout for diagnostic clarity."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, f'subprocess failed: {exc!r}'
    return result.returncode, result.stdout + result.stderr


def _pytest_collect_count():
    """Return ``(pass=None, collected, skip=None, xfail=None)`` from a
    collect-only run.

    Skip/xfail breakdown is NOT available from collect-only -- the
    counts come from the marker pre-screen, not from running the tests.
    The script's ``--quick`` mode therefore only refreshes the ``Y
    collected`` field and derives ``X pass = collected - skip - xfail``
    (holding the at-write-time skip / xfail counts constant).  Users
    wanting an empirical pass/skip/xfail breakdown pass ``--full``.

    Return-tuple shape matches ``_pytest_full_count``: ``(pass_n,
    coll_n, skip_n, xfail_n)``.  Quick mode leaves the pass / skip /
    xfail slots as None to signal "use the in-block at-write-time
    values"; only ``coll_n`` is empirical.
    """
    if not _TESTS_UNIT.is_dir():
        return None, None, None, None
    rc, out = _run(
        [sys.executable, '-m', 'pytest', str(_TESTS_UNIT),
         '-m', 'not integration', '--collect-only', '-q',
         '-p', 'no:cacheprovider'],
        timeout=120,
    )
    if rc != 0 and rc != 5:
        # rc=5 is "no tests collected" which is genuine; other nonzero
        # is a collection failure (import error etc.) that we treat as
        # "unavailable" rather than "zero tests".
        return None, None, None, None
    # pytest's collect-only quiet output ends with one of:
    #     "N tests collected in T.TTs"
    #     "no tests collected"
    # The exact format has been stable since pytest 6.x.
    for line in out.splitlines()[::-1]:
        line = line.strip()
        m = re.search(r'(\d+)\s+tests?\s+collected', line)
        if m:
            return None, int(m.group(1)), None, None
    return None, None, None, None


def _pytest_full_count():
    """Return ``(pass, collected, skip, xfail)`` from a full suite run.

    Slow (~5 minutes on a typical workstation).  Parses pytest's
    summary line which looks like:
        "===== 3848 passed, 17 skipped, 1 xfailed in 145.2s ====="
    Order of categories is not guaranteed; each is matched independently.
    """
    if not _TESTS_UNIT.is_dir():
        return None, None, None, None
    rc, out = _run(
        [sys.executable, '-m', 'pytest', str(_TESTS_UNIT),
         '-m', 'not integration', '-q',
         '-p', 'no:cacheprovider'],
        timeout=600,
    )
    pass_n = skip_n = xfail_n = 0
    # Walk from the bottom (the summary line is near the end).
    for line in out.splitlines()[::-1]:
        if '====' not in line:
            continue
        m = re.search(r'(\d+)\s+passed', line)
        if m:
            pass_n = int(m.group(1))
        m = re.search(r'(\d+)\s+skipped', line)
        if m:
            skip_n = int(m.group(1))
        m = re.search(r'(\d+)\s+xfailed', line)
        if m:
            xfail_n = int(m.group(1))
        if pass_n > 0:
            break
    if pass_n == 0:
        return None, None, None, None
    collected = pass_n + skip_n + xfail_n
    return pass_n, collected, skip_n, xfail_n


def _git_changed_file_count(prev_tag, head_ref='HEAD'):
    """Return the file-count from ``git diff PREV_TAG..HEAD --name-only``.

    Returns None when git is unavailable or the diff fails.
    """
    rc, out = _run(['git', 'diff', '--name-only',
                    f'{prev_tag}..{head_ref}'],
                   timeout=30)
    if rc != 0:
        return None
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return len(files)


def _git_net_loc(prev_tag, head_ref='HEAD'):
    """Return ``(insertions, deletions)`` from
    ``git diff PREV_TAG..HEAD --shortstat``.

    The git-shortstat output looks like:
        " 54 files changed, 13813 insertions(+), 1075 deletions(-)"

    Either side can be missing (a release that only adds OR only
    deletes lines).  Missing sides default to 0 -- this matches
    git's own convention.

    Returns ``(None, None)`` when git is unavailable, the diff
    fails, or the shortstat line cannot be parsed.
    """
    rc, out = _run(['git', 'diff', '--shortstat',
                    f'{prev_tag}..{head_ref}'],
                   timeout=30)
    if rc != 0:
        return None, None
    # The shortstat output is a single line (possibly with a
    # trailing newline).  Be defensive and walk all non-empty
    # lines -- some git versions emit a blank leading line.
    for line in out.splitlines():
        line = line.strip()
        if 'changed' not in line:
            continue
        ins_match = re.search(r'(\d+)\s+insertion', line)
        del_match = re.search(r'(\d+)\s+deletion', line)
        ins = int(ins_match.group(1)) if ins_match else 0
        del_ = int(del_match.group(1)) if del_match else 0
        # If both sides are zero AND the regex didn't fire, the
        # parse failed; signal None to skip the stamp.
        if ins_match is None and del_match is None:
            return None, None
        return ins, del_
    return None, None


def _resolve_prev_tag(target_version):
    """Find the previous git tag to use as the diff base.

    Strategy:
      1. List ``v*`` tags ordered by version (``git tag -l 'v*'
         --sort=-version:refname``).
      2. Pick the highest tag strictly less than ``v<target_version>``.
      3. If the target is the current in-progress release (no tag yet),
         the highest existing tag IS the previous one.
      4. Return None if no usable tag exists.
    """
    rc, out = _run(['git', 'tag', '-l', 'v*',
                    '--sort=-version:refname'], timeout=10)
    if rc != 0:
        return None
    target_tag = f'v{target_version}'
    tags = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not tags:
        return None
    for tag in tags:
        if tag != target_tag:
            return tag
    return None


def _file_line_count(path):
    """Return the line count for ``path`` (mirrors ``wc -l``)."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    # wc -l counts newline characters; a file without a trailing
    # newline has one fewer "line" than splitlines() reports.
    return text.count('\n')


# ---------------------------------------------------------------------------
# Rewrite engine
# ---------------------------------------------------------------------------

class _StampPlan:
    """Container for the proposed stamps + the resulting rewritten body."""

    def __init__(self):
        self.changes = []   # list of (label, old, new) tuples
        self.skipped = []   # list of label strings -- no pattern found


def _stamp_test_counts(body, pass_n, coll_n, skip_n, xfail_n, mode):
    """Rewrite test-count claims in ``body``.

    Returns ``(new_body, change_or_None)``.  ``change`` is a
    ``(label, old, new)`` tuple suitable for the diff preview.
    """
    match = _TEST_COUNT_PATTERN.search(body)
    if match is None:
        return body, None
    old_pass = int(match.group('pass'))
    old_coll = int(match.group('coll'))
    old_skip = int(match.group('skip'))
    old_xfail = int(match.group('xfail'))

    if mode == 'quick':
        # ``--quick`` only knows the collected total.  Hold skip/xfail
        # constant and compute pass = collected - skip - xfail.
        new_coll = coll_n
        new_skip = old_skip
        new_xfail = old_xfail
        new_pass = new_coll - new_skip - new_xfail
        if new_pass < 0:
            # The collected count went DOWN below the stamped skip+xfail
            # sum.  Refuse to write a nonsense pass count; treat as no-op.
            return body, None
    else:
        # ``--full`` knows all four; honor them.
        new_pass = pass_n
        new_coll = coll_n
        new_skip = skip_n
        new_xfail = xfail_n

    if (old_pass, old_coll, old_skip, old_xfail) == (
            new_pass, new_coll, new_skip, new_xfail):
        # Already correct; nothing to do.
        return body, None

    # Rebuild the matched span with new numbers, preserving the
    # original whitespace / glue captured in m1..m4.
    new_span = (
        f'{new_pass}{match.group("m1")}{new_coll}{match.group("m2")}'
        f'{new_skip}{match.group("m3")}{new_xfail}{match.group("m4")}'
    )
    new_body = body[:match.start()] + new_span + body[match.end():]
    label = (
        f'test counts: {old_pass} pass / {old_coll} coll / {old_skip} '
        f'skip / {old_xfail} xfail  ->  {new_pass} / {new_coll} / '
        f'{new_skip} / {new_xfail}'
    )
    return new_body, (label, match.group(0), new_span)


def _stamp_file_count(body, n_files):
    """Rewrite ``N files touched|changed|modified`` claims.

    Returns ``(new_body, change_or_None)``.
    """
    if n_files is None:
        return body, None
    match = _FILE_COUNT_PATTERN.search(body)
    if match is None:
        return body, None
    old_n = int(match.group('n'))
    if old_n == n_files:
        return body, None
    new_span = f'{n_files}{match.group("m")}'
    new_body = body[:match.start()] + new_span + body[match.end():]
    label = f'file count: {old_n} -> {n_files} (git diff PREV_TAG..HEAD)'
    return new_body, (label, match.group(0), new_span)


def _stamp_line_count(body, n_lines):
    """Rewrite the CHANGELOG.md line-count claim.

    Returns ``(new_body, change_or_None)``.  Looks for both the
    ``OLD -> NEW lines`` form and the ``from OLD to NEW lines`` form.
    Only the post-arrow / post-"to" number is updated; the
    pre-archive number is left intact (it is a historical anchor).
    """
    if n_lines is None:
        return body, None
    for pattern in (_LINE_COUNT_ARROW_PATTERN, _LINE_COUNT_FROM_TO_PATTERN):
        match = pattern.search(body)
        if match is None:
            continue
        old_n = int(match.group('n'))
        if old_n == n_lines:
            return body, None
        # Patterns have 3 unnamed groups + the named 'n'; rebuild.
        new_span = f'{match.group(1)}{n_lines}{match.group(3)}'
        new_body = body[:match.start()] + new_span + body[match.end():]
        label = f'CHANGELOG.md line count: {old_n} -> {n_lines}'
        return new_body, (label, match.group(0), new_span)
    return body, None


def _stamp_net_loc(body, loc_ins, loc_del):
    """Rewrite ``Net LOC: +X / -Y [vs vA.B.C]`` claims.

    Added in v5.4.1 (AUDIT_V5_4_0 P2 #3) to catch the deletions-side
    drift class: V17.2's tolerance band only covers the ``+`` axis,
    and the v5.4.0 Phase 6 tombstone deletion (-619 lines) slipped
    past unnoticed.  Stamping both numbers from ``--shortstat``
    closes the loop.

    Returns ``(new_body, change_or_None)``.  When the empirical
    pair matches the cited pair, the body is returned unchanged.
    """
    if loc_ins is None or loc_del is None:
        return body, None
    match = _NET_LOC_PATTERN.search(body)
    if match is None:
        return body, None
    old_ins = int(match.group('ins'))
    old_del = int(match.group('del_'))
    if (old_ins, old_del) == (loc_ins, loc_del):
        return body, None
    new_span = (
        f'{match.group("pre")}{loc_ins}{match.group("mid")}'
        f'{loc_del}{match.group("suf")}'
    )
    new_body = body[:match.start()] + new_span + body[match.end():]
    label = (
        f'Net LOC: +{old_ins} / -{old_del}  ->  '
        f'+{loc_ins} / -{loc_del} (git diff PREV_TAG..HEAD --shortstat)'
    )
    return new_body, (label, match.group(0), new_span)


def _build_plan(body, mode, pass_n, coll_n, skip_n, xfail_n,
                file_count, line_count, loc_ins=None, loc_del=None):
    """Apply all rewrites to ``body`` and return a ``_StampPlan``."""
    plan = _StampPlan()
    new_body = body

    if coll_n is not None:
        new_body, change = _stamp_test_counts(
            new_body, pass_n, coll_n, skip_n, xfail_n, mode)
        if change is None:
            if _TEST_COUNT_PATTERN.search(body) is None:
                plan.skipped.append('test counts: no pattern to update')
        else:
            plan.changes.append(change)
    else:
        plan.skipped.append('test counts: empirical collection unavailable')

    new_body, change = _stamp_file_count(new_body, file_count)
    if change is None:
        if file_count is None:
            plan.skipped.append('file count: PREV_TAG not resolvable')
        elif _FILE_COUNT_PATTERN.search(body) is None:
            plan.skipped.append('file count: no pattern to update')
    else:
        plan.changes.append(change)

    new_body, change = _stamp_line_count(new_body, line_count)
    if change is None:
        # Either no claim in block, or already matches.
        has_pattern = (
            _LINE_COUNT_ARROW_PATTERN.search(body) is not None
            or _LINE_COUNT_FROM_TO_PATTERN.search(body) is not None
        )
        if not has_pattern:
            plan.skipped.append('line count: no pattern to update')
    else:
        plan.changes.append(change)

    # v5.4.1 addition: Net LOC: +X / -Y stamp.  Order it last so the
    # diff-preview reads top-to-bottom as the patterns appear in the
    # canonical "### Files touched" subsection of a CHANGELOG entry.
    new_body, change = _stamp_net_loc(new_body, loc_ins, loc_del)
    if change is None:
        if loc_ins is None or loc_del is None:
            plan.skipped.append(
                'Net LOC: PREV_TAG not resolvable or shortstat failed')
        elif _NET_LOC_PATTERN.search(body) is None:
            plan.skipped.append('Net LOC: no pattern to update')
    else:
        plan.changes.append(change)

    plan.new_body = new_body
    return plan


def _render_diff(old_text, new_text, label):
    """Render a unified diff between two text strings."""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f'{label} (before)',
        tofile=f'{label} (after)',
        n=2,
    )
    return ''.join(diff)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def stamp(version=None, apply_changes=False, mode='quick', stream=sys.stdout):
    """Build the stamp plan; print it; optionally apply.

    Returns the exit code (0/1/2/3 per the module docstring).
    """
    text = _read_changelog()

    ver, body, start, end = _find_version_block(text, version)
    if ver is None:
        if version is None:
            print('ERROR: CHANGELOG.md has no ``## [X.Y.Z]`` blocks; '
                  'nothing to stamp.  (exit 3)', file=stream)
        else:
            print(f'ERROR: CHANGELOG.md has no ``## [{version}]`` '
                  f'block.  (exit 3)', file=stream)
        return 3

    print(f'stamp_changelog: targeting v{ver} block, mode={mode}, '
          f'apply={apply_changes}', file=stream)

    # Gather empirical counts.
    print('  collecting empirical counts ...', file=stream)
    if mode == 'full':
        pass_n, coll_n, skip_n, xfail_n = _pytest_full_count()
    else:
        pass_n, coll_n, skip_n, xfail_n = _pytest_collect_count()
    if coll_n is not None:
        print(f'    pytest ({mode}): collected={coll_n}'
              + (f', pass={pass_n}, skip={skip_n}, xfail={xfail_n}'
                 if mode == 'full' else ''),
              file=stream)
    else:
        print(f'    pytest ({mode}): unavailable (tests/unit/ missing '
              f'or pytest failed)', file=stream)

    prev_tag = _resolve_prev_tag(ver)
    if prev_tag is None:
        print('    git tag PREV: unresolvable (no prior v* tag found)',
              file=stream)
        file_count = None
        loc_ins = None
        loc_del = None
    else:
        file_count = _git_changed_file_count(prev_tag)
        if file_count is None:
            print(f'    git diff {prev_tag}..HEAD: failed', file=stream)
        else:
            print(f'    git diff {prev_tag}..HEAD: {file_count} files',
                  file=stream)
        # v5.4.1 addition: --shortstat for the Net LOC stamp.
        loc_ins, loc_del = _git_net_loc(prev_tag)
        if loc_ins is None or loc_del is None:
            print(
                f'    git diff {prev_tag}..HEAD --shortstat: failed',
                file=stream)
        else:
            print(
                f'    git diff {prev_tag}..HEAD --shortstat: '
                f'+{loc_ins} / -{loc_del}',
                file=stream)

    line_count = _file_line_count(_CHANGELOG)
    if line_count is not None:
        print(f'    CHANGELOG.md line count: {line_count}', file=stream)

    plan = _build_plan(
        body, mode, pass_n, coll_n, skip_n, xfail_n,
        file_count, line_count, loc_ins, loc_del,
    )

    # Report.
    print(file=stream)
    if not plan.changes and not plan.skipped:
        print(f'  no stampable patterns in v{ver} block; nothing to do.  '
              f'(exit 2)', file=stream)
        return 2

    if plan.skipped:
        print('  skipped patterns:', file=stream)
        for note in plan.skipped:
            print(f'    - {note}', file=stream)

    if not plan.changes:
        print('  no drift detected -- all stampable patterns already '
              'current.  (exit 0)', file=stream)
        return 0

    print('  proposed stamps:', file=stream)
    for label, _, _ in plan.changes:
        print(f'    - {label}', file=stream)

    # Build the proposed new CHANGELOG.md.
    new_text = text[:start] + plan.new_body + text[end:]
    diff_text = _render_diff(text, new_text, 'CHANGELOG.md')
    print(file=stream)
    print('  unified diff preview:', file=stream)
    for line in diff_text.splitlines():
        print(f'    {line}', file=stream)

    if apply_changes:
        _CHANGELOG.write_text(new_text, encoding='utf-8')
        print(file=stream)
        print(f'  --apply: wrote {len(plan.changes)} stamp(s) to '
              f'{_CHANGELOG}.  (exit 0)', file=stream)
        return 0

    print(file=stream)
    print(f'  dry-run: {len(plan.changes)} stamp(s) ready to write; '
          f'pass --apply to commit.  (exit 1)', file=stream)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Stamp CHANGELOG.md self-citation patterns with ship-time '
            'empirical values.  Dry-run by default; pass --apply to '
            'actually rewrite.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true',
        help=('Actually rewrite CHANGELOG.md.  Without this flag the '
              'script prints a diff preview and exits without writing.'),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--quick', action='store_const', const='quick', dest='mode',
        default='quick',
        help=('Use ``pytest --collect-only`` for the test count (fast, '
              '~5 seconds).  Default mode.'),
    )
    mode_group.add_argument(
        '--full', action='store_const', const='full', dest='mode',
        help=('Run the full unit suite to refresh pass/skip/xfail '
              'counts.  Slow (~5 minutes).'),
    )
    parser.add_argument(
        '--version', '-V', default=None,
        help=('Target a specific past version block (e.g. ``5.2.3``).  '
              'Default: topmost block.'),
    )
    args = parser.parse_args(argv)

    return stamp(
        version=args.version,
        apply_changes=args.apply,
        mode=args.mode,
    )


if __name__ == '__main__':
    sys.exit(main())
