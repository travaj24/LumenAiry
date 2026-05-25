"""V19 walker -- scope-the-workaround comment pattern.

Detects comment patterns indicating scoped workarounds in shipped
library code:

  - ``# v5.\\d+ candidate``
  - ``# scoped workaround``
  - ``# scope.*workaround`` (case-insensitive)
  - ``# workaround.*v5\\.\\d+`` (e.g. "workaround until v5.5")
  - ``# TODO.*v5\\.\\d+``
  - ``# defer.*to.*v5\\.\\d+``
  - ``# .*real.*fix.*lives`` (audit-cited "real fix lives elsewhere"
    phrasing)

For each match, this walker requires a paired ``ROADMAP.md`` entry
that references the file:line OR topic.  Without a paired entry the
workaround is invisible to the audit cadence and risks being lost
between releases.

Background.  v5.4.0 shipped the ``_ghost_intersect`` workaround in
``analysis/ghost.py`` with a CHANGELOG-honest comment naming the
deferred real fix as a "v5.5 candidate".  The
``AUDIT_V5_4_0_2026_05_25.md`` audit (Part 6 + Part 7 #13)
classified this as a NEW meta-pattern (k=4): real library bug
discovered, scoped workaround in the consumer layer, deferred to a
future release.  CHANGELOG-honest, code-deferred.  v5.4.1 (Wave 1A)
promoted the ``_ghost_intersect`` fix into the canonical
``_intersect_surface`` site, but the meta-pattern protection is
still required for the NEXT scoped workaround that ships without a
paired tracker.

V19 closes the k=4 sibling.  Walker is necessary-but-not-sufficient:
it does NOT verify the ROADMAP entry actually tracks the workaround,
only that one exists.  Augmenting with the audit-cycle review remains
essential.

Test contract:
    test_v19_no_unpaired_scope_the_workaround_comments_in_library
        Canonical check.  Scans all ``lumenairy/**/*.py`` for the
        audit-named patterns; for each match, requires the
        ``ROADMAP.md`` text to mention the file:line, the file path,
        or the file basename.  Unpaired findings fail with a
        actionable error message.
    test_v19_synthetic_workaround_without_roadmap_pairing_is_caught
        Walker self-test: constructs an in-memory synthetic finding
        pointing at a non-existent file path that ROADMAP cannot
        reference; asserts the pairing logic flags it.  Defends
        against a future regex/pairing-logic regression that would
        silently pass everything.
    test_v19_finds_at_least_the_documented_v5_4_examples
        Diagnostic harness: runs the scanner against the live
        library and asserts it returns a finite list, printing the
        findings for ``pytest -v`` inspection.

Closes ROADMAP v5.4.1 audit P3 item #13 (V19 walker for k=4
scope-the-workaround meta-pattern).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LUMENAIRY_DIR = _REPO_ROOT / 'lumenairy'
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'


# Comment patterns that indicate a scoped workaround.  Tuned to the
# audit-named phrasings (AUDIT_V5_4_0_2026_05_25.md Part 6 + Part 7
# #13); generic ``# TODO`` comments are intentionally NOT included --
# they require the ``v5.\d+`` version suffix to trigger so the
# walker does not become a TODO-policing nuisance.
_WORKAROUND_PATTERNS = [
    (re.compile(r'#.*v5\.\d+\s+candidate', re.IGNORECASE),
     'v5.N candidate'),
    (re.compile(r'#.*scoped\s+workaround', re.IGNORECASE),
     'scoped workaround'),
    (re.compile(r'#.*workaround.*v5\.\d+', re.IGNORECASE),
     'versioned workaround'),
    (re.compile(r'#.*TODO.*v5\.\d+', re.IGNORECASE),
     'versioned TODO'),
    (re.compile(r'#.*defer.*to.*v5\.\d+', re.IGNORECASE),
     'versioned deferral'),
    (re.compile(r'#.*real\s+fix\s+lives', re.IGNORECASE),
     'real-fix-lives-elsewhere'),
]


# Files / directories to skip.  Library code only -- UI, tests,
# scripts, and examples all carry their own version markers that
# aren't library-correctness workarounds.
_SKIP_DIRS = {'ui', '__pycache__'}


def _scan_library_for_workaround_comments():
    """Return list of ``(file:line, comment_text, pattern_kind)``
    tuples for every scope-the-workaround comment in library code.

    Library code = ``lumenairy/**/*.py`` minus the directories named
    in ``_SKIP_DIRS``.  At most one finding is emitted per source
    line (first pattern wins).
    """
    findings = []
    for py_file in _LUMENAIRY_DIR.rglob('*.py'):
        # Skip excluded subdirs.
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        try:
            text = py_file.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, kind in _WORKAROUND_PATTERNS:
                if pattern.search(line):
                    rel_path = py_file.relative_to(_REPO_ROOT).as_posix()
                    findings.append(
                        (f'{rel_path}:{lineno}', line.strip(), kind))
                    break  # one match per line
    return findings


def _roadmap_text():
    return _ROADMAP.read_text(encoding='utf-8', errors='replace')


def _is_paired(file_line, roadmap_lower):
    """Return True if ``roadmap_lower`` references ``file_line``, its
    file path, or its basename stem.

    Pairing is lenient on purpose: V19 is necessary-but-not-sufficient.
    Any of the three references (full ``file:line``, the rel-path
    alone, or the basename stem) signals an intent to track the
    deferral.  An audit-cycle reviewer then verifies the ROADMAP
    item actually tracks THIS workaround.
    """
    file_part = file_line.split(':', 1)[0]
    basename = Path(file_part).stem
    return (
        file_line.lower() in roadmap_lower
        or file_part.lower() in roadmap_lower
        or basename.lower() in roadmap_lower
    )


# ===========================================================================
# V19.1 -- canonical check: every scope-the-workaround comment is paired
# ===========================================================================

def test_v19_no_unpaired_scope_the_workaround_comments_in_library():
    """V19: every scope-the-workaround comment in library code must
    have a paired ROADMAP entry that references the file or topic.

    Catches the v5.4.0 k=4 meta-pattern surfaced in
    ``AUDIT_V5_4_0_2026_05_25.md`` Part 6.  Without this pin, scoped
    workarounds risk being lost between audit cycles even when the
    CHANGELOG comment is honest about the deferred real fix.
    """
    findings = _scan_library_for_workaround_comments()
    roadmap = _roadmap_text().lower()

    unpaired = []
    for file_line, comment_text, kind in findings:
        if not _is_paired(file_line, roadmap):
            unpaired.append((file_line, comment_text, kind))

    assert not unpaired, (
        f'V19: found {len(unpaired)} scope-the-workaround comment(s) '
        f'in library code without paired ROADMAP entries.  Each '
        f'scoped workaround must have a corresponding ROADMAP item '
        f'that tracks the deferred-real-fix; otherwise the workaround '
        f'risks being lost between audit cycles (k=4 meta-pattern '
        f'from v5.4.0 audit, AUDIT_V5_4_0_2026_05_25.md Part 6).\n\n'
        f'Unpaired findings:\n'
        + '\n'.join(
            f'  {fl}  [{kind}]:  {ct}' for fl, ct, kind in unpaired)
        + '\n\nFix options:\n'
        '  (a) promote the workaround into the real fix site '
        '(closes the deferral entirely),\n'
        '  (b) add a ROADMAP entry tracking the deferral '
        '(references the file:line, file path, or basename), or\n'
        '  (c) reword the comment to drop the "v5.N candidate" / '
        '"scoped workaround" / "real fix lives" framing if the '
        'comment is not actually announcing a deferred fix.')


# ===========================================================================
# V19.2 -- self-test: walker catches an unpaired synthetic finding
# ===========================================================================

def test_v19_synthetic_workaround_without_roadmap_pairing_is_caught():
    """Synthetic regression: construct a fake unpaired finding and
    assert the pairing logic flags it.

    Uses an in-memory mock rather than touching the file system, so
    the test never plants a real workaround comment in the tree.
    Defends against a future regression in either the regex set or
    the pairing predicate that would silently pass every finding.
    """
    # Path that ROADMAP.md definitely does not reference -- the
    # ``_synthetic`` subtree does not exist in the repo and is not
    # mentioned in any ROADMAP item.
    synthetic_findings = [
        ('lumenairy/_synthetic/nonexistent_walker_v19.py:42',
         '# v5.5 candidate: real fix lives in some_other_module',
         'v5.N candidate'),
    ]
    roadmap = _roadmap_text().lower()

    unpaired = [
        finding for finding in synthetic_findings
        if not _is_paired(finding[0], roadmap)
    ]
    assert unpaired, (
        'V19 self-test: synthetic unpaired workaround was not caught; '
        'the regex / pairing logic is broken.  Either the synthetic '
        'path leaked into ROADMAP.md (rename the sentinel) or '
        '``_is_paired`` is now too permissive (e.g. matching empty '
        'string).')


# ===========================================================================
# V19.3 -- self-test: walker catches each documented pattern kind
# ===========================================================================

def test_v19_each_pattern_kind_matches_its_canonical_example():
    """Companion self-test for the regex set.  Each of the six
    pattern kinds must match its canonical phrasing exactly; a
    refactor that accidentally narrowed a pattern would otherwise
    silently miss the meta-pattern it was added to catch.
    """
    canonical_examples = [
        ('# v5.5 candidate: promote this fix to intersection.py',
         'v5.N candidate'),
        ('# scoped workaround in the consumer layer',
         'scoped workaround'),
        ('# workaround until v5.5 rolls out the shared-state fix',
         'versioned workaround'),
        ('# TODO(v5.6): replace this with the canonical implementation',
         'versioned TODO'),
        ('# defer to v5.5 for the proper canonical implementation',
         'versioned deferral'),
        ('# the real fix lives in raytrace/intersection.py',
         'real-fix-lives-elsewhere'),
    ]
    for example, expected_kind in canonical_examples:
        matched_kind = None
        for pattern, kind in _WORKAROUND_PATTERNS:
            if pattern.search(example):
                matched_kind = kind
                break
        assert matched_kind == expected_kind, (
            f'V19 self-test: canonical example {example!r} did not '
            f'match expected pattern kind {expected_kind!r}; got '
            f'{matched_kind!r} instead.  The regex set has drifted '
            f'from the audit-named phrasings and will silently miss '
            f'the k=4 meta-pattern.')


# ===========================================================================
# V19.4 -- diagnostic harness: scanner returns a finite list
# ===========================================================================

def test_v19_finds_at_least_the_documented_v5_4_examples(capsys):
    """v5.4.0 documented several scoped-workaround patterns
    (``_ghost_intersect`` in ``analysis/ghost.py``, plus secondary
    versioned-workaround comments elsewhere).  By v5.4.1 the primary
    one is promoted but secondary patterns may remain.  This test
    runs the scanner against the live library and asserts the
    scanner mechanism works -- returns a finite list, didn't crash.
    Prints the findings for ``pytest -v`` diagnostic value.
    """
    findings = _scan_library_for_workaround_comments()
    # Don't assert exact count -- count varies as fixes land.  Just
    # assert the scanner returns a finite list of tuples.
    assert isinstance(findings, list)
    for entry in findings:
        assert isinstance(entry, tuple) and len(entry) == 3, (
            f'V19 scanner returned a malformed finding entry: '
            f'{entry!r}; expected (file:line, comment, kind) tuple.')

    # Diagnostic output -- ``pytest -v`` will surface this.
    print(f'\nV19 walker found {len(findings)} scope-the-workaround '
          f'comment(s) in library code:')
    for file_line, comment_text, kind in findings:
        snippet = comment_text if len(comment_text) <= 80 \
            else comment_text[:77] + '...'
        print(f'  {file_line}  [{kind}]:  {snippet}')
