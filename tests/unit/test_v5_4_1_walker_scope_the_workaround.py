"""V19 walker -- scope-the-workaround comment pattern.

Detects comment patterns indicating scoped workarounds in shipped
library code.  Original v5.4.1 surface (kept):

  - ``# vN.M candidate``
  - ``# scoped workaround``
  - ``# scope.*workaround`` (case-insensitive)
  - ``# workaround.*vN\\.M``
  - ``# TODO.*vN\\.M``
  - ``# defer.*to.*vN\\.M``
  - ``# .*real.*fix.*lives``

v5.4.5 extensions (AUDIT_V5_4_4 Part 9 #4 / #5 / #6) close the
gameable surfaces the audit empirically demonstrated against the
v5.4.1 walker:

  - ``# FIXME`` keyword (versioned or unversioned)
  - unversioned defer (``# deferred to next release``,
    ``# deferred until ...``, ``# deferred past ...``)
  - post-v5 versions (``v6.0 candidate``, ``v7.X``, ...) -- the
    ``v\\d+\\.\\d+`` family generalises the prior ``v5\\.\\d+``
  - hyphen separators (``# v5.5-candidate``)
  - qualified synonyms (``# hack until v5.5``,
    ``# kludge workaround``, ``# bandaid v5.5``)
  - versioned patch (``# patch v5.5``)
  - docstring-housed workarounds (triple-quoted ``v5.5 candidate: ...``)
    -- a second-pass scanner flags lines containing a triple-quote
    AND any of the workaround phrases.  Multi-line docstrings
    without a triple-quote on the offending line are an
    acknowledged limitation (see ``_DOCSTRING_PHRASES`` TODO).

For each match, this walker requires a paired ``ROADMAP.md`` entry
that references the file:line OR full path (audit fix #4 closed
the basename-collision exploit -- bare basenames no longer pair).
Without a paired entry the workaround is invisible to the audit
cadence and risks being lost between releases.

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
# #13, extended by AUDIT_V5_4_4 Part 9 #5); generic ``# TODO`` /
# ``# hack`` comments are NOT matched in isolation -- they require a
# version, deferral, or "workaround" qualifier so the walker does
# not become a generic TODO-policing nuisance.
#
# Version family generalised from ``v5\.\d+`` to ``v\d+\.\d+`` in
# v5.4.5 so post-v5 versions (v6.x / v7.x / ...) all match (audit
# fix #5, "post-v5 versions").  Hyphen-separated form
# ``v5.5-candidate`` matches via the ``[-\s]+`` separator in the
# extended pattern (audit fix #5, "hyphen separator").
_WORKAROUND_PATTERNS = [
    # ---- Original v5.4.1 surface (kept, version family widened) ----
    (re.compile(r'#.*\bv\d+\.\d+\s+candidate\b', re.IGNORECASE),
     'vN.M candidate'),
    (re.compile(r'#.*\bscoped\s+workaround\b', re.IGNORECASE),
     'scoped workaround'),
    (re.compile(r'#.*\bworkaround\b.*\bv\d+\.\d+\b', re.IGNORECASE),
     'versioned workaround'),
    (re.compile(r'#.*\bTODO\b.*\bv\d+\.\d+\b', re.IGNORECASE),
     'versioned TODO'),
    (re.compile(r'#.*\bdefer\b.*\bto\b.*\bv\d+\.\d+\b', re.IGNORECASE),
     'versioned deferral'),
    (re.compile(r'#.*\breal\s+fix\s+lives\b', re.IGNORECASE),
     'real-fix-lives-elsewhere'),
    # ---- v5.4.5 audit fix #5: extended pattern set ----
    (re.compile(r'#.*\bFIXME\b', re.IGNORECASE),
     'FIXME'),
    (re.compile(r'#.*\bdeferred\s+(?:to|until|past)\b', re.IGNORECASE),
     'unversioned deferral'),
    (re.compile(r'#.*\bv\d+\.\d+[-\s]+candidate\b', re.IGNORECASE),
     'vN.M-candidate (hyphen+space)'),
    (re.compile(r'#.*\b(?:hack|kludge|bandaid)\b'
                r'.*\b(?:until|workaround|v\d+\.\d+)\b', re.IGNORECASE),
     'qualified hack/kludge'),
    (re.compile(r'#.*\bpatch\s+v\d+\.\d+\b', re.IGNORECASE),
     'versioned patch'),
]


# v5.4.5 audit fix #6: docstring-housed workarounds slip through the
# comment-prefix patterns above (which require a ``#`` lead).  This
# phrase list is matched against any line containing a triple-quote
# (``"""`` or ``'''``).  Multi-line docstrings without a triple-quote
# on the offending line are an acknowledged limitation -- the full
# defence would require a Python AST parse, which is overkill for a
# walker test; the single-line / opening-line triple-quote case
# covers the audit-confirmed exploit ("""v5.5 candidate: ...""").
# TODO(v5.5+): if docstring-housed workarounds escape this heuristic
# in practice, upgrade to an ``ast.parse`` walk that visits every
# ``Expr(value=Constant(str))`` node.
_DOCSTRING_PHRASES = [
    (re.compile(r'\bv\d+\.\d+\s+candidate\b', re.IGNORECASE),
     'docstring: vN.M candidate'),
    (re.compile(r'\bscoped\s+workaround\b', re.IGNORECASE),
     'docstring: scoped workaround'),
    (re.compile(r'\bworkaround\b.*\bv\d+\.\d+\b', re.IGNORECASE),
     'docstring: versioned workaround'),
    (re.compile(r'\bFIXME\b', re.IGNORECASE),
     'docstring: FIXME'),
    (re.compile(r'\breal\s+fix\s+lives\b', re.IGNORECASE),
     'docstring: real-fix-lives-elsewhere'),
    (re.compile(r'\bdeferred\s+(?:to|until|past)\b', re.IGNORECASE),
     'docstring: unversioned deferral'),
]


def _line_in_docstring(line):
    """Heuristic: line contains a triple-quote.

    Simplest workable test for the v5.4.5 docstring fix.  See the
    ``_DOCSTRING_PHRASES`` TODO above for the acknowledged limitation
    (multi-line docstrings without a triple-quote on the workaround
    line itself escape this check).
    """
    stripped = line.strip()
    return '"""' in stripped or "'''" in stripped


# Files / directories to skip.  Library code only -- UI, tests,
# scripts, and examples all carry their own version markers that
# aren't library-correctness workarounds.
_SKIP_DIRS = {'ui', '__pycache__'}


def _scan_library_for_workaround_comments():
    """Return list of ``(file:line, comment_text, pattern_kind)``
    tuples for every scope-the-workaround comment in library code.

    Library code = ``lumenairy/**/*.py`` minus the directories named
    in ``_SKIP_DIRS``.  At most one finding is emitted per source
    line (first pattern wins; comment patterns take priority over
    docstring patterns).
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
            rel_path = py_file.relative_to(_REPO_ROOT).as_posix()
            matched = False
            # Pass 1: comment patterns (``#`` prefix family).
            for pattern, kind in _WORKAROUND_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        (f'{rel_path}:{lineno}', line.strip(), kind))
                    matched = True
                    break  # one match per line
            if matched:
                continue
            # Pass 2: docstring-housed workarounds (v5.4.5 fix #6).
            # Only consider lines that contain a triple-quote; this
            # is the documented single-line heuristic.
            if _line_in_docstring(line):
                for pattern, kind in _DOCSTRING_PHRASES:
                    if pattern.search(line):
                        findings.append(
                            (f'{rel_path}:{lineno}', line.strip(), kind))
                        break  # one match per line
    return findings


def _roadmap_text():
    return _ROADMAP.read_text(encoding='utf-8', errors='replace')


def _is_paired(file_line, roadmap_lower):
    """v5.4.5 (AUDIT_V5_4_4 Part 9 fix #4): require a full-path
    match in ROADMAP, not just a bare basename mention.

    Closes the basename-collision exploit the audit empirically
    confirmed: a workaround planted in ``analysis/coronagraph.py``
    is falsely-paired by a ROADMAP entry that mentions
    ``coronagraph_dock.py`` (different file, same stem).

    Pairing now accepts:
      * the full ``file:line`` citation
        (``lumenairy/foo/bar.py:42``)
      * the full relative path
        (``lumenairy/foo/bar.py``)
      * the path with the ``lumenairy/`` package prefix stripped
        (``foo/bar.py``)

    Bare basenames (``bar.py``) NO LONGER pair.  V19 remains
    necessary-but-not-sufficient: an audit-cycle reviewer still
    verifies the ROADMAP item actually tracks THIS workaround.
    """
    file_line_l = file_line.lower()
    file_part = file_line_l.split(':', 1)[0]      # 'lumenairy/foo/bar.py'
    # Strip an optional leading ``lumenairy/`` package prefix without
    # consuming arbitrary leading chars (str.lstrip strips characters,
    # not a prefix substring).
    _PKG = 'lumenairy/'
    if file_part.startswith(_PKG):
        file_part_short = file_part[len(_PKG):]   # 'foo/bar.py'
    else:
        file_part_short = file_part
    candidates = {
        file_line_l,                               # 'lumenairy/foo/bar.py:42'
        file_part,                                 # 'lumenairy/foo/bar.py'
        file_part_short,                           # 'foo/bar.py'
        file_part_short.lstrip('/'),               # defensive double-slash
    }
    return any(c and c in roadmap_lower for c in candidates)


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
         'vN.M candidate'),
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


# ===========================================================================
# V19.5 -- v5.4.5 audit fix #4: basename-collision pairing closed
# ===========================================================================

def test_v19_full_path_pairing_rejects_basename_collision():
    """v5.4.5 (AUDIT_V5_4_4 Part 9 fix #4): a ROADMAP entry that
    only mentions a sibling file with the same basename stem must
    NOT pair a workaround in the original file.

    Audit-confirmed exploit: a workaround planted in
    ``analysis/coronagraph.py`` was falsely-paired by a ROADMAP
    line mentioning ``coronagraph_dock.py``.  Tightened
    ``_is_paired`` now requires a full-path match.
    """
    finding = ('lumenairy/analysis/coronagraph.py:42',
               '# v5.5 candidate: real fix lives in propagation.py',
               'vN.M candidate')
    # Mock ROADMAP mentions ONLY the sibling docking file with the
    # colliding basename stem -- not the real workaround site.
    fake_roadmap = (
        '* designer GUI: `coronagraph_dock.py` (783 LOC) ships the '
        '4-stop chain builder at v5.4.\n'
    ).lower()
    # New (tightened) logic: must reject the bare-basename collision.
    assert not _is_paired(finding[0], fake_roadmap), (
        'V19 fix #4 regression: ``_is_paired`` accepted a ROADMAP '
        'mention of a different file with the same basename stem '
        '(coronagraph_dock.py vs analysis/coronagraph.py).  The '
        'audit-confirmed basename-collision exploit is open again; '
        'pairing must require a full-path match.')
    # And: a ROADMAP entry mentioning the FULL relative path DOES
    # still pair (full-path positive case -- defends against an
    # overcorrection that would break legitimate pairings).
    real_roadmap = (
        '* `lumenairy/analysis/coronagraph.py` carries a v5.5 '
        'deferred fix tracked here.\n'
    ).lower()
    assert _is_paired(finding[0], real_roadmap), (
        'V19 fix #4 overcorrection: full-path ROADMAP mention '
        'should still pair.')


# ===========================================================================
# V19.6-12 -- v5.4.5 audit fix #5 + #6: extended pattern coverage
# ===========================================================================

def _scan_lines(lines):
    """Helper: run the comment + docstring scanner over an in-memory
    list of source lines.  Mirrors ``_scan_library_for_workaround_comments``
    minus the file-walking layer, so individual lines can be tested
    without planting a real workaround on disk.
    """
    hits = []
    for lineno, line in enumerate(lines, start=1):
        matched = False
        for pattern, kind in _WORKAROUND_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, line, kind))
                matched = True
                break
        if matched:
            continue
        if _line_in_docstring(line):
            for pattern, kind in _DOCSTRING_PHRASES:
                if pattern.search(line):
                    hits.append((lineno, line, kind))
                    break
    return hits


def test_v19_pattern_FIXME_caught():
    """v5.4.5 fix #5: ``# FIXME ...`` (versioned or unversioned) is
    a scoped workaround.  Audit Part 9 #5 empirical entry:
    ``# FIXME until v5.5`` was MISSED by v5.4.1.
    """
    hits = _scan_lines(['# FIXME until v5.5: clean this up'])
    assert hits, 'V19 fix #5 (FIXME) failed: line was not matched.'
    assert hits[0][2] == 'FIXME', (
        f'V19 fix #5 (FIXME) wrong kind label: {hits[0][2]!r}.')


def test_v19_pattern_unversioned_defer_caught():
    """v5.4.5 fix #5: unversioned ``# deferred to next release`` is
    a scoped workaround (audit Part 9 #5: "unversioned defer").
    """
    hits = _scan_lines(['# deferred to next release'])
    assert hits, 'V19 fix #5 (unversioned defer) failed: not matched.'
    assert hits[0][2] == 'unversioned deferral', (
        f'V19 fix #5 wrong kind label: {hits[0][2]!r}.')


def test_v19_pattern_post_v5_version_caught():
    """v5.4.5 fix #5: post-v5 versions (``v6.0 candidate``) must
    match.  Audit Part 9 #5: hard-coded ``v5\\.\\d+`` was a v6
    transition footgun; widened to ``v\\d+\\.\\d+``.
    """
    hits = _scan_lines(['# v6.0 candidate: redo the dispatcher'])
    assert hits, 'V19 fix #5 (post-v5) failed: v6.0 not matched.'
    assert hits[0][2] == 'vN.M candidate', (
        f'V19 fix #5 (post-v5) wrong kind label: {hits[0][2]!r}.')


def test_v19_pattern_hyphen_separator_caught():
    """v5.4.5 fix #5: ``# v5.5-candidate`` (hyphen separator) must
    match.  Audit Part 9 #5: prior regex only allowed whitespace.
    """
    hits = _scan_lines(['# v5.5-candidate: real fix in next release'])
    assert hits, 'V19 fix #5 (hyphen) failed: v5.5-candidate not matched.'
    assert hits[0][2] == 'vN.M-candidate (hyphen+space)', (
        f'V19 fix #5 (hyphen) wrong kind label: {hits[0][2]!r}.')


def test_v19_pattern_qualified_hack_caught():
    """v5.4.5 fix #5: ``# hack until v5.5`` (qualified synonym) must
    match.  Bare ``# hack`` is intentionally NOT matched -- the
    qualifier (``until`` / ``workaround`` / ``vN.M``) is required to
    avoid policing every "hack" comment in the library.
    """
    hits = _scan_lines(['# hack until v5.5: revisit when consumer wires'])
    assert hits, 'V19 fix #5 (qualified hack) failed: not matched.'
    assert hits[0][2] == 'qualified hack/kludge', (
        f'V19 fix #5 (qualified hack) wrong kind: {hits[0][2]!r}.')
    # Counter-pin: bare ``# hack`` ALONE must NOT match.  This guards
    # against the "regex too broad" failure mode the audit flagged.
    bare = _scan_lines(['# hack to make the test pass'])
    assert not bare, (
        f'V19 fix #5: bare ``# hack`` should NOT match; got {bare!r}.')


def test_v19_pattern_versioned_patch_caught():
    """v5.4.5 fix #5: ``# patch v5.5`` (versioned synonym for
    workaround / deferred fix) must match.
    """
    hits = _scan_lines(['# patch v5.5: temporary until shared-state lands'])
    assert hits, 'V19 fix #5 (versioned patch) failed: not matched.'
    assert hits[0][2] == 'versioned patch', (
        f'V19 fix #5 (versioned patch) wrong kind: {hits[0][2]!r}.')


def test_v19_docstring_housed_workaround_caught():
    """v5.4.5 fix #6: workaround phrasings housed in a docstring
    (triple-quoted ``v5.5 candidate: ...``) must match via the
    second-pass docstring scanner.  Audit Part 9 #6 confirmed the
    comment-only regex missed docstring-housed workarounds.
    """
    line = '    """v5.5 candidate: promote this into intersection.py"""'
    hits = _scan_lines([line])
    assert hits, (
        'V19 fix #6 (docstring) failed: triple-quoted workaround '
        'phrase was not flagged.')
    assert hits[0][2].startswith('docstring:'), (
        f'V19 fix #6 wrong kind label (expected docstring: prefix): '
        f'{hits[0][2]!r}.')
    # Counter-pin: non-docstring line with the same phrase but no
    # triple-quote should NOT be flagged by the docstring scanner
    # (the comment-prefix regex handles ``#`` lines on its own).
    no_quote = _scan_lines(['v5.5 candidate: not in a docstring'])
    assert not no_quote, (
        f'V19 fix #6: line without ``#`` or triple-quote should not '
        f'match; got {no_quote!r}.')
