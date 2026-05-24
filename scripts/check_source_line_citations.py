#!/usr/bin/env python3
# v5.3.2 (ROADMAP V18 walker -- source-file:line citation drift):
"""check_source_line_citations.py -- CHANGELOG source-file:line citation walker.

Companion CLI for the V18 walker
(``tests/unit/test_v5_3_2_walker_source_line_citation.py``).  V12 catches
FILE-LEVEL citation fabrications (the cited file does not exist).  V16
catches CONTENT-LEVEL fabrications (the file exists but the cited change
is not in the diff).  V18 closes a third sibling: LINE-LEVEL drift in
source-file:line citations.

The v5.3.0 ship hit a real instance of this failure mode.  v5.3.0's
CHANGELOG cited ``optimize/wrapper_merits.py:855`` for the
``_ZERO_APERTURE_MASK`` sentinel branch.  The v5.3.0 MultiFieldMerit
JIT change added ~19 LOC above the sentinel branch in the same file,
shifting the cited site to line 876.  The
``test_v4_15_agent_f.py::TestF5ChangelogLineCitations`` pin caught it,
but only because that pin has its own anchor-based source-locator for
a few specific symbols.  V18 is the GENERAL version of the pin:  it
parses every backticked ``lumenairy/foo/bar.py:N`` citation in the
topmost CHANGELOG block and verifies that line N still exists AND is
"non-trivial" (the structural cue that the cited symbol is still
near).

Heuristic: a NON-TRIVIAL line is a line that is NOT just whitespace,
NOT a closing bracket, and NOT a bare ``pass`` / ``continue`` /
``break``.  A non-trivial line at the cited position is the structural
anchor; if the cited line is trivial, the citation is probably stale.

This is a NECESSARY-BUT-NOT-SUFFICIENT check.  V18 does not try to
verify the cited SYMBOL is at the cited line (that requires an
anchor-string database, which ``test_v4_15_agent_f.py`` builds by hand
for specific symbols).  V18 just verifies "the cited line still exists
AND is non-trivial."

Usage:
    python scripts/check_source_line_citations.py
        -- audit the topmost ``## [X.Y.Z]`` block

    python scripts/check_source_line_citations.py --version 5.3.0
        -- audit a specific past version's block

    python scripts/check_source_line_citations.py --quiet
        -- suppress per-citation output; only print summary

Exit codes:
    0  -- all source-file:line citations land on non-trivial lines (OK)
    1  -- one or more citations land on trivial lines OR out-of-range
          (CITATION DRIFT)
    2  -- no source-file:line citations to audit (skip cleanly)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'


# Source-file:line citation pattern.  Matches backticked tokens of the
# form:
#   * ``lumenairy/foo/bar.py:N`` (or ``:START-END``)         -- full path
#   * ``tests/unit/test_foo.py:N`` (or ``:START-END``)       -- full path
#   * ``foo.py:N`` (or ``:START-END``)                       -- bare basename
#
# Bare-basename citations are common in CHANGELOG prose where the bullet
# context makes the directory obvious (e.g. "``mhs.py:573-611``" inside a
# bullet about ``propagate_huygens_fresnel_freespace``).  V18 resolves
# bare basenames against the repo: a basename with EXACTLY ONE match
# under ``lumenairy/`` / ``tests/`` / ``scripts/`` / ``examples/`` is
# treated as a real citation; ambiguous basenames are skipped (a
# basename that resolves to multiple files cannot be line-checked).
#
# The .md / .toml / .yml file types are deliberately excluded -- those
# class of citations (e.g. ``CHANGELOG.md:81``) are V17's domain.
_SOURCE_LINE_PATTERN = re.compile(
    r'`(?P<path>'
    r'(?:lumenairy|tests|examples|scripts|benchmarks|validation)'
    r'/[\w\-./]+\.py'
    r'|\.github/[\w\-./]+\.py'
    r'|[\w\-]+\.py'
    r')'
    r':(?P<start>\d+)(?:-(?P<end>\d+))?'
    r'`')


# Trivial-line heuristics.  A "trivial" line is one that is unlikely to
# be the canonical anchor a CHANGELOG bullet would have cited.
_TRIVIAL_TOKENS = ('pass', 'continue', 'break', '...', 'else:', 'try:')


def _is_trivial_line(line):
    """Return True if ``line`` is a trivial / closing-bracket line.

    A non-trivial line is the structural anchor; a trivial line at the
    cited position usually means the citation has drifted.
    """
    stripped = line.strip()
    if not stripped:
        return True
    # Pure closing-bracket / structural-only lines.
    if stripped in (')', ']', '}', '),', '],', '},'):
        return True
    # Comment-only lines that say nothing structural (a bare ``#`` or
    # ``# ---`` divider).  Comments with content ARE non-trivial.
    if stripped.startswith('#'):
        # Strip leading ``#`` + whitespace and check for content.
        comment_body = stripped.lstrip('#').strip()
        if not comment_body or set(comment_body) <= set('-=*_'):
            return True
        # Otherwise comments are non-trivial (they often anchor V18
        # citations -- the v5.3.0 ROADMAP comments are line-cited).
        return False
    # Bare keyword statements with no payload.
    if stripped in _TRIVIAL_TOKENS:
        return True
    if stripped.startswith('pass ') or stripped.startswith('return\n'):
        return True
    return False


# ---------------------------------------------------------------------------
# CHANGELOG parsing -- mirrors V12 / V16 / V17 walker idiom
# ---------------------------------------------------------------------------

def _read_changelog():
    if not _CHANGELOG.is_file():
        raise SystemExit(
            f'ERROR: CHANGELOG.md not found at {_CHANGELOG}; nothing '
            f'to audit.  (exit 2)')
    return _CHANGELOG.read_text(encoding='utf-8')


def _all_version_blocks(text):
    """Return list of (version, body, start) for every ``## [X.Y.Z]``
    block in CHANGELOG.md, top-to-bottom.
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
    """Pick the target block by version (or the topmost block if None)."""
    if not blocks:
        return None
    if version is None:
        return blocks[0]
    for blk in blocks:
        if blk[0] == version:
            return blk
    return None


def _extract_citations(body):
    """Yield (path, start_line, end_line_or_None) tuples for every
    backticked source-file:line citation in ``body``.

    Citations of the form ``path:N`` produce ``(path, N, None)``;
    ``path:START-END`` produce ``(path, START, END)``.
    """
    out = []
    for m in _SOURCE_LINE_PATTERN.finditer(body):
        path = m.group('path')
        start = int(m.group('start'))
        end = int(m.group('end')) if m.group('end') else None
        out.append((path, start, end))
    return out


# Cached basename -> path resolution.  Built lazily on first lookup.
_BASENAME_INDEX_CACHE = None


def _build_basename_index():
    """Return a {basename: [paths]} map for every ``.py`` file under
    the repo's tracked top-level dirs.

    Used to resolve bare-basename citations (e.g. ``mhs.py:573-611``).
    """
    global _BASENAME_INDEX_CACHE
    if _BASENAME_INDEX_CACHE is not None:
        return _BASENAME_INDEX_CACHE
    index = {}
    search_roots = (
        _REPO_ROOT / 'lumenairy',
        _REPO_ROOT / 'tests',
        _REPO_ROOT / 'scripts',
        _REPO_ROOT / 'examples',
        _REPO_ROOT / 'benchmarks',
        _REPO_ROOT / 'validation',
    )
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.rglob('*.py'):
            # Skip __pycache__ + egg-info etc.
            if '__pycache__' in p.parts:
                continue
            if any(part.endswith('.egg-info') for part in p.parts):
                continue
            rel = p.relative_to(_REPO_ROOT).as_posix()
            index.setdefault(p.name, []).append(rel)
    _BASENAME_INDEX_CACHE = index
    return index


def _resolve_path(cited_path):
    """Resolve a cited path string to a real repo-relative path.

    For full-path citations (contains ``/``), returns the path as-is.
    For bare-basename citations (e.g. ``mhs.py``), looks up the
    basename index and returns the unique match if there is exactly
    one; returns None if zero or more-than-one matches.
    """
    if '/' in cited_path:
        return cited_path
    index = _build_basename_index()
    matches = index.get(cited_path, [])
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Per-citation verification
# ---------------------------------------------------------------------------

def _verify_citation(path, start_line, end_line):
    """Verify a single ``path:start[-end]`` citation.

    Returns a 3-tuple ``(status, detail, sample)`` where:
      * ``status`` is one of ``'ok'``, ``'missing-file'``,
        ``'ambiguous-basename'``, ``'out-of-range'``, ``'trivial-line'``.
      * ``detail`` is a short human-readable explanation.
      * ``sample`` is the cited line text (or empty when not applicable).

    Note: ``'ambiguous-basename'`` is treated as a clean skip (not a
    failure) by the audit -- if a bare basename matches multiple files,
    V18 cannot decide which one the bullet meant.
    """
    resolved = _resolve_path(path)
    if resolved is None:
        if '/' in path:
            return 'missing-file', f'{path} not present in repo', ''
        # Bare basename with zero or multiple matches.
        index = _build_basename_index()
        matches = index.get(path, [])
        if not matches:
            return ('missing-file',
                    f'bare basename ``{path}`` matches no file in repo',
                    '')
        return ('ambiguous-basename',
                f'bare basename ``{path}`` matches {len(matches)} '
                f'files: {matches!r}; V18 cannot disambiguate', '')

    p = _REPO_ROOT / resolved
    if not p.is_file():
        return 'missing-file', f'{resolved} not present in repo', ''

    try:
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as exc:
        return 'missing-file', f'{path}: OSError {exc!r}', ''

    n = len(lines)
    if start_line < 1 or start_line > n:
        return ('out-of-range',
                f'{path}: cited line {start_line} but file has {n} '
                f'lines', '')

    # Check a 5-line window centered on the cited line.  If EITHER the
    # cited line itself OR every line in the +/- 2 window is trivial,
    # the citation is stale.  The asymmetric requirement reflects the
    # v5.3.0 retrospective: the actual sentinel-branch line itself was
    # the anchor that drifted; checking the cited line directly is the
    # primary signal.  The window check is the secondary fallback.
    cited = lines[start_line - 1]
    if not _is_trivial_line(cited):
        return 'ok', f'cited line {start_line} is non-trivial', cited.strip()

    # Cited line is trivial.  Scan the +/- 2 window for ANY non-trivial
    # neighbour; if all five lines are trivial, the citation is almost
    # certainly stale (no anchor in the immediate vicinity).
    lo = max(0, start_line - 3)
    hi = min(n, start_line + 2)
    window = lines[lo:hi]
    any_non_trivial = any(not _is_trivial_line(ln) for ln in window)
    if any_non_trivial:
        # The cited line itself is trivial but the local neighbourhood
        # has structural content -- a soft drift.  Still flagged but
        # with a softer detail message.
        return ('trivial-line',
                f'{path}:{start_line} lands on a trivial line '
                f'(``{cited.strip()!r}``); a non-trivial neighbour '
                f'exists in the 5-line window, but the cited line '
                f'itself has likely drifted', cited.strip())

    return ('trivial-line',
            f'{path}:{start_line} lands on a trivial line and the '
            f'entire 5-line window is structurally empty',
            cited.strip())


# ---------------------------------------------------------------------------
# Public API used by the walker
# ---------------------------------------------------------------------------

def audit_block(version=None, quiet=False):
    """Audit CHANGELOG block's source-file:line citations.

    Returns one of:
        0  -- all citations land on non-trivial lines (OK)
        1  -- one or more citations are stale (drift)
        2  -- no citations to audit (skip cleanly)
    """
    text = _read_changelog()
    blocks = _all_version_blocks(text)
    target = _select_block(blocks, version)

    if target is None:
        if version is None:
            print('ERROR: CHANGELOG.md has no ``## [X.Y.Z]`` blocks.')
        else:
            print(f'ERROR: CHANGELOG.md has no ``## [{version}]`` block.')
        return 2

    target_version, target_body, _ = target

    if not quiet:
        print(f'check_source_line_citations: auditing v{target_version} '
              f'block.')

    citations = _extract_citations(target_body)
    if not citations:
        if not quiet:
            print(f'  no source-file:line citations in v{target_version} '
                  f'block; nothing to verify.  (exit 2)')
        return 2

    failures = []
    skipped_ambig = 0
    ok_count = 0

    if not quiet:
        print(f'  found {len(citations)} citation(s)')
        print()
        print('  STATUS              CITATION -- detail')
        print('  ------              ' + '-' * 60)

    for path, start, end in citations:
        status, detail, sample = _verify_citation(path, start, end)
        cite_str = f'{path}:{start}'
        if end is not None:
            cite_str += f'-{end}'

        if status == 'ok':
            ok_count += 1
            if not quiet:
                preview = sample[:50] if sample else ''
                print(f'  OK                  {cite_str}  ``{preview}``')
        elif status == 'ambiguous-basename':
            skipped_ambig += 1
            if not quiet:
                print(f'  SKIP-AMBIGUOUS      {cite_str}')
                print(f'                      -- {detail}')
        else:
            failures.append((cite_str, status, detail))
            if not quiet:
                print(f'  {status.upper():<19} {cite_str}')
                print(f'                      -- {detail}')

    if not quiet:
        print()
        print(f'  summary: ok={ok_count}  drift={len(failures)}  '
              f'skip_ambig={skipped_ambig}  total={len(citations)}')

    if failures:
        if quiet:
            print(f'CITATION DRIFT in v{target_version}:')
            for cite_str, status, detail in failures:
                print(f'  - {cite_str}  [{status}]  {detail}')
        return 1

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Verify CHANGELOG source-file:line citations have not drifted. '
            'Line-level companion to V12 / V16 / V17 walkers.'
        ),
    )
    parser.add_argument(
        '--version', '-V', default=None,
        help=('Audit a specific past version block (e.g. ``5.3.0``).  '
              'Default: topmost block.'),
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress per-citation OK output; print only failures + summary.',
    )
    args = parser.parse_args(argv)

    rc = audit_block(version=args.version, quiet=args.quiet)
    return rc


if __name__ == '__main__':
    sys.exit(main())
