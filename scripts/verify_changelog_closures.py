#!/usr/bin/env python3
# v5.2.3 (AUDIT_V5_1_0 P1-NEW-2WAY-1 + V12 companion content-level walker):
"""verify_changelog_closures.py -- content-level CHANGELOG fabrication walker.

Companion to the V12 file-existence walker
(``tests/unit/test_v5_2_walker_changelog_changeset.py``).  V12 catches
FILE-LEVEL fabrications: a backticked file-path citation that resolves
to a non-existent file, or an audit-ID citation that resolves to no
audit document.  V12 deliberately does NOT cover CONTENT-LEVEL
fabrications -- the file exists but the cited behavior is missing.
v5.1.0's ``publish.yml verify gate`` fabrication was that exact class:
the file existed (publish.yml has been there since v3.x); the new
``verify`` job described in the bullet was absent from the file.

This script closes the content-level gap by diffing the cited paths
against the ``git diff PREV_TAG..HEAD`` changeset.  A bullet may cite
``tests/unit/foo.py`` only if ``foo.py`` actually appears in the diff;
otherwise the bullet is flagged as a candidate fabrication.

Run manually at tag time AND wired into ``.github/workflows/publish.yml``
verify job so a release with fabricated closures cannot upload to PyPI.

Usage:
    python scripts/verify_changelog_closures.py
        -- audit the topmost ``## [X.Y.Z]`` block, diff vs previous
           version block's tag

    python scripts/verify_changelog_closures.py --version 5.2.0
        -- audit a specific past version's block

    python scripts/verify_changelog_closures.py --strict
        -- also check audit-ID resolution (V12.2 redundancy)

    python scripts/verify_changelog_closures.py --quiet
        -- suppress the per-bullet OK table; only print failures

Exit codes:
    0  -- all cited file paths have backing diffs (OK)
    1  -- one or more cited file paths missing from the diff
          (CANDIDATE FABRICATION)
    2  -- no git history / no PREV_TAG resolvable (skip cleanly)
    3  -- CHANGELOG.md not parseable, requested version not present,
          or other input error
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants / regexes -- borrowed from V12 walker, then specialised
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_AUDITS_DIR = _REPO_ROOT / 'docs' / 'audits'


# Path tokens that look like real repo files; same shape as V12
# walker's _FILE_PATH_PATTERN, but here we also allow leading
# ``./`` for paste-style citations.
_FILE_PATH_PATTERN = re.compile(
    r'`(?P<path>'
    r'(?:lumenairy|tests|examples|docs|scripts|benchmarks|validation)'
    r'/[\w\-./]+\.\w{1,5}'
    r'|\.github/[\w\-./]+\.\w{1,5}'
    r'|(?:README|CHANGELOG|ROADMAP|Migration-Guide|CONTRIBUTING|CONVENTIONS)\.md'
    r'|pyproject\.toml'
    r'|requirements\.txt'
    r')'
    r'(?P<lines>:\d+(?:-\d+)?)?'
    r'`')


_AUDIT_ID_PATTERN = re.compile(
    r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z][A-Z0-9_-]{2,})')


# Bullet markers at the start of a stripped line.
_BULLET_START = re.compile(r'^[\*\-]\s+')


# Heuristics for "this bullet is doc-only / non-code, skip diff check":
# - bullet text mentions only documentation files (``.md``, ``CONVENTIONS``)
# - bullet text contains "stub" / "deferred" / "no library API change"
# - bullet text contains "documentation" or "docstring" without a code-file
#   citation
_DOC_ONLY_HINTS = (
    'documentation only',
    'no library api change',
    'data only',
    'deferred to',
    'historical accuracy',
    'preserved verbatim',
)


# ---------------------------------------------------------------------------
# CHANGELOG parsing -- mirrors V12 walker idiom
# ---------------------------------------------------------------------------

def _read_changelog():
    if not _CHANGELOG.is_file():
        raise SystemExit(
            f'ERROR: CHANGELOG.md not found at {_CHANGELOG}; nothing '
            f'to audit.  (exit 3)')
    return _CHANGELOG.read_text(encoding='utf-8')


def _all_version_blocks(text):
    """Return list of (version_string, body_text, start_offset) for every
    ``## [X.Y.Z]`` block in CHANGELOG.md, ordered top-to-bottom.

    Body runs from after the heading line to the next ``## `` heading
    (or end-of-file).
    """
    blocks = []
    pattern = re.compile(
        r'^## \[(?P<ver>\d+\.\d+\.\d+)\][^\n]*\n(?P<body>.*?)'
        r'(?=^## \[|\Z)',
        re.DOTALL | re.MULTILINE)
    for match in pattern.finditer(text):
        blocks.append((match.group('ver'), match.group('body'),
                       match.start()))
    return blocks


def _select_block(blocks, version):
    """Return (target_block, prev_block) for the requested version.

    If ``version`` is None, the target is the topmost block and prev
    is the second block.  If ``version`` is given, the target is the
    matching block and prev is the next-older block.

    Either may be None if no such block exists.
    """
    if not blocks:
        return None, None
    if version is None:
        target = blocks[0]
        prev = blocks[1] if len(blocks) > 1 else None
        return target, prev
    for i, blk in enumerate(blocks):
        if blk[0] == version:
            target = blk
            prev = blocks[i + 1] if i + 1 < len(blocks) else None
            return target, prev
    return None, None


# ---------------------------------------------------------------------------
# Audit-closure bullet extraction
# ---------------------------------------------------------------------------

def _is_audit_closure_section(heading_line):
    """Return True if the ``### `` heading announces an audit-closure
    section.  Matches:

        ### v5.1.1 audit closures (5 items)
        ### v5.1.0 audit carry-over (1 item)
        ### P1 closures (3)
        ### P2 closures (5)
        ### P3 closures (8)
        ### audit closures
        ### audit P3 fix
        ### New audit P3 fix
        ### Tier 1 closures (5 substantive residuals)        [v5.2.5]
        ### Tier 2 closures (2 documentation refactors)       [v5.2.5]
        ### Tier 3 closures (3 tools / infra)                 [v5.2.5]
        ### Tier 1 -- substantive residuals                   [v5.2.5]
        ### v5.2.0 audit closures actually shipped in v5.2.5  [v5.2.5]

    v5.2.5 (AUDIT_V5_2_3 P2-F1-3): extended to recognize "Tier N
    closures" (v5.2.3 used that form for its 3 closure sections;
    the v5.2.3 release's verify_changelog_closures run reported
    rc=2 "no audit-closure bullets detected" because the
    classifier missed the Tier-N heading family).  Future
    audit-closure heading variants should be added here, NOT
    silently allowed past the gate.
    """
    s = heading_line.lower()
    return (
        'audit closure' in s
        or 'audit carry-over' in s
        or 'audit carryover' in s
        or 'audit fix' in s
        or re.search(r'\bp[0-3]\s+closure', s) is not None
        # v5.2.5 (AUDIT_V5_2_3 P2-F1-3): Tier N closures form
        or re.search(r'\btier\s+[0-9]\b', s) is not None
    )


def _extract_audit_closure_bullets(body):
    """Walk the body line-by-line; collect bullets that live under an
    ``### `` heading whose text indicates an audit-closure section.

    Returns a list of bullet-text strings (one bullet per entry, with
    its continuation lines joined into one string).
    """
    bullets = []
    in_closure_section = False
    current_bullet = None
    current_section = None

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # New ### heading?
        if line.startswith('### '):
            # Flush any pending bullet
            if current_bullet is not None:
                bullets.append((current_section, current_bullet))
                current_bullet = None
            heading_text = line[4:]
            current_section = heading_text
            in_closure_section = _is_audit_closure_section(heading_text)
            continue

        # New ## or # heading exits the section entirely
        if line.startswith('## ') or line.startswith('# '):
            if current_bullet is not None:
                bullets.append((current_section, current_bullet))
                current_bullet = None
            in_closure_section = False
            continue

        if not in_closure_section:
            continue

        # Within a closure section.  A new top-level bullet?
        # Top-level bullets start at column 0 with ``*`` or ``-``.
        if _BULLET_START.match(line):
            if current_bullet is not None:
                bullets.append((current_section, current_bullet))
            current_bullet = stripped
            continue

        # Continuation of current bullet?  Indented line + non-empty.
        if current_bullet is not None and line.startswith(' '):
            current_bullet += ' ' + stripped
            continue

        # Blank line or de-indent without a bullet marker -- flush.
        if current_bullet is not None and not stripped:
            bullets.append((current_section, current_bullet))
            current_bullet = None
            continue

    if current_bullet is not None:
        bullets.append((current_section, current_bullet))

    return bullets


def _extract_cited_paths(bullet_text):
    """Yield (path, line_spec_or_None) for every backticked file path
    cited in the bullet text.  Strips the optional ``:N`` or ``:N-M``
    line suffix off the path component but preserves it in the second
    tuple element for diagnostic output."""
    paths = []
    for m in _FILE_PATH_PATTERN.finditer(bullet_text):
        paths.append((m.group('path'), m.group('lines') or ''))
    return paths


def _extract_cited_audit_ids(bullet_text):
    return [m.group(1) for m in _AUDIT_ID_PATTERN.finditer(bullet_text)]


def _is_doc_only_bullet(bullet_text):
    """Heuristic: bullet is documentation / metadata-only and shouldn't
    be diff-checked.  Triggers if the bullet text contains a doc-only
    hint phrase OR cites only ``.md`` files."""
    low = bullet_text.lower()
    if any(hint in low for hint in _DOC_ONLY_HINTS):
        return True
    paths = _extract_cited_paths(bullet_text)
    if not paths:
        # No code-file citations at all -- can't be diff-checked
        # without a target.  We treat that as doc-only for purposes
        # of the OK/MISSING table.
        return True
    if all(p.endswith('.md') for p, _ in paths):
        return True
    return False


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------

def _run_git(args):
    """Run ``git <args>`` from the repo root.  Returns (returncode,
    stdout_text).  Captures stderr to stdout for diagnostic clarity."""
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, f'git unavailable: {exc!r}'
    return result.returncode, result.stdout


def _resolve_prev_tag(prev_block, target_version):
    """Determine the PREV_TAG to diff against.

    Strategy:
      1. If a previous CHANGELOG block exists, use ``v<prev_version>``.
      2. Fall back to ``git describe --tags --abbrev=0 v<target>^``
         if the target's tag exists.
      3. Return None if neither works.
    """
    candidates = []
    if prev_block is not None:
        candidates.append(f'v{prev_block[0]}')

    # Verify each candidate exists as a tag
    for cand in candidates:
        rc, _ = _run_git(['rev-parse', '--verify', f'refs/tags/{cand}'])
        if rc == 0:
            return cand

    # Fall back: parent of the target tag, if the target tag exists
    target_tag = f'v{target_version}'
    rc, _ = _run_git(['rev-parse', '--verify', f'refs/tags/{target_tag}'])
    if rc == 0:
        rc2, out2 = _run_git(['describe', '--tags', '--abbrev=0',
                              f'{target_tag}^'])
        if rc2 == 0 and out2.strip():
            return out2.strip()

    return None


def _resolve_head_ref(target_version, target_is_latest):
    """Pick the HEAD-side ref for the diff.

    If the target is a past version with a corresponding tag, diff
    against that tag (so we audit the ship state, not the current
    working tree).  Otherwise, diff against ``HEAD`` (the in-progress
    release).
    """
    target_tag = f'v{target_version}'
    rc, _ = _run_git(['rev-parse', '--verify', f'refs/tags/{target_tag}'])
    if rc == 0 and not target_is_latest:
        return target_tag
    if rc == 0:
        # If the latest CHANGELOG block matches an existing tag, we
        # still prefer the tag -- it reflects the ship state.
        return target_tag
    return 'HEAD'


def _changed_files(prev_tag, head_ref):
    """Return set of paths changed between prev_tag and head_ref."""
    rc, out = _run_git(['diff', '--name-only', f'{prev_tag}..{head_ref}'])
    if rc != 0:
        return None
    return set(line.strip() for line in out.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_block(version=None, strict=False, quiet=False):
    """Verify CHANGELOG block's audit-closure bullets against git diff.

    Returns one of:
        0  -- all cited file paths backed by diff
        1  -- at least one cited file path missing from diff
        2  -- skipped cleanly (no git history, no audit closures, etc.)
        3  -- input error (version not found, CHANGELOG unparseable)
    """
    text = _read_changelog()
    blocks = _all_version_blocks(text)
    target, prev = _select_block(blocks, version)

    if target is None:
        if version is None:
            print('ERROR: CHANGELOG.md has no ``## [X.Y.Z]`` blocks; '
                  'nothing to verify.  (exit 3)')
        else:
            print(f'ERROR: CHANGELOG.md has no ``## [{version}]`` '
                  f'block.  (exit 3)')
        return 3

    target_version, target_body, _ = target
    target_is_latest = (blocks[0][0] == target_version)

    if not quiet:
        print(f'verify_changelog_closures: auditing v{target_version} '
              f'block ({"latest" if target_is_latest else "past"}).')

    bullets = _extract_audit_closure_bullets(target_body)
    if not bullets:
        if not quiet:
            print(f'  no audit-closure bullets detected in v{target_version} '
                  f'block; nothing to verify.  (exit 2)')
        return 2

    prev_tag = _resolve_prev_tag(prev, target_version)
    if prev_tag is None:
        if not quiet:
            print(f'  no PREV_TAG resolvable (prev block: '
                  f'{prev[0] if prev else "(none)"}); skipping content '
                  f'check.  (exit 2)')
        return 2

    head_ref = _resolve_head_ref(target_version, target_is_latest)
    changed = _changed_files(prev_tag, head_ref)
    if changed is None:
        if not quiet:
            print(f'  git diff {prev_tag}..{head_ref} failed; skipping.  '
                  f'(exit 2)')
        return 2

    if not quiet:
        print(f'  diff range: {prev_tag}..{head_ref}  '
              f'({len(changed)} files changed)')
        print()
        print('  STATUS    SECTION/BULLET'
              ' (first ~70 chars) -- cited paths')
        print('  ------    ' + '-' * 70)

    failures = []
    skipped_doc_only = 0
    ok_count = 0
    unresolved_audit_ids = []

    for section, bullet_text in bullets:
        cited = _extract_cited_paths(bullet_text)
        preview = bullet_text[:70].replace('\n', ' ')

        if _is_doc_only_bullet(bullet_text):
            status = 'SKIP-DOC'
            skipped_doc_only += 1
            if not quiet:
                print(f'  {status}  {preview!r}')
            continue

        missing_paths = []
        for path, lines in cited:
            if path.endswith('.md'):
                # doc-only path; skip in mixed bullets too
                continue
            if path not in changed:
                missing_paths.append(f'{path}{lines}')

        if missing_paths:
            status = 'MISSING'
            failures.append({
                'section': section,
                'preview': preview,
                'missing': missing_paths,
                'cited': [p for p, _ in cited],
            })
            if not quiet:
                print(f'  {status}   {preview!r}')
                for mp in missing_paths:
                    print(f'             -- not in diff: {mp}')
        else:
            status = 'OK'
            ok_count += 1
            if not quiet:
                cited_str = ', '.join(p for p, _ in cited[:3])
                if len(cited) > 3:
                    cited_str += f', +{len(cited) - 3} more'
                print(f'  {status}        {preview!r}'
                      f'  [{cited_str}]')

        if strict:
            for aid in _extract_cited_audit_ids(bullet_text):
                if not _audit_id_resolves(aid):
                    unresolved_audit_ids.append(aid)

    if not quiet:
        print()
        print(f'  summary: ok={ok_count}  missing={len(failures)}  '
              f'skip_doc={skipped_doc_only}  '
              f'total_bullets={len(bullets)}')
        if strict:
            print(f'  strict-mode: unresolved audit IDs = '
                  f'{len(unresolved_audit_ids)}')

    if failures:
        if quiet:
            # Quiet mode prints only the failures
            print(f'FABRICATION CANDIDATES in v{target_version}:')
            for f in failures:
                print(f'  - section: {f["section"]}')
                print(f'    bullet:  {f["preview"]!r}')
                print(f'    missing: {f["missing"]}')
        return 1

    if strict and unresolved_audit_ids:
        if not quiet:
            print(f'  strict-mode failure: unresolved audit IDs: '
                  f'{sorted(set(unresolved_audit_ids))}')
        return 1

    return 0


def _audit_id_resolves(audit_id):
    """True if audit_id appears under docs/audits/."""
    if not _AUDITS_DIR.is_dir():
        return True  # can't verify; don't penalise
    for f in _AUDITS_DIR.rglob('*.md'):
        try:
            if audit_id in f.read_text(encoding='utf-8', errors='replace'):
                return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Verify CHANGELOG audit-closure bullets have backing diffs. '
            'Content-level companion to the V12 file-existence walker.'
        ),
    )
    parser.add_argument(
        '--version', '-V', default=None,
        help=('Audit a specific past version block (e.g. ``5.2.0``).  '
              'Default: topmost block.'),
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Also verify every cited audit ID resolves under docs/audits/.',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress per-bullet OK output; print only failures + summary.',
    )
    args = parser.parse_args(argv)

    rc = verify_block(
        version=args.version,
        strict=args.strict,
        quiet=args.quiet,
    )
    return rc


if __name__ == '__main__':
    sys.exit(main())
