"""Structural meta-pin for the ``np.where(..., 0.0 + 0.0j)`` anti-pattern.

Audit context
-------------

``AUDIT_V4_14_1_2026_05_17.md`` P1-NEW-4 surfaced a recurring code
anti-pattern: ``np.where(mask, expr, 0.0 + 0.0j)``.  The literal
``0.0 + 0.0j`` is a Python ``complex128`` object, so broadcasting it
through ``np.where`` silently upcasts a ``complex64`` ``expr`` to
``complex128`` -- doubling the memory footprint and breaking the
v4.10+ memory-budgeting contract for users who set
``set_default_complex_dtype(np.complex64)``.

The canonical replacement is dtype-aware::

    xp.where(mask, E, xp.zeros((), dtype=E.dtype))

which produces a 0-D array of the correct precision, preserving the
caller's dtype across the broadcast.

History
-------

The v4.13.2 sweep fixed several apply-X sites but missed (per audit
3-way confirmation V2 + F1 + F2):

* ``optimize/core.py:966`` (Agent B's scope this release)
* ``analysis/phase_retrieval.py:402`` (Agent C's scope this release)
* ``analysis/phase_retrieval.py:367`` -- has trailing
  ``.astype(cdtype)`` recovery so dtype-correctness is restored
  downstream; rated **P3** in the audit.
* ``ui/psf_mtf_dock.py:230`` -- UI code, rated **P3** in the audit.

This pin walks the ``lumenairy/`` source tree, regex-matches the
anti-pattern on each line, and post-filters lines that have a trailing
``.astype(`` recovery on the same line (the P3 sites above).  Any
match without recovery is a regression -- the test fails and prints
the file:line for triage.

After Agent B and Agent C land their v4.14.2 fixes, this pin should
pass.  A future regression that lands a new bare ``np.where(...,
0.0 + 0.0j)`` (no astype recovery) will trip it.

Robustness notes
----------------

* **Scope:** ``lumenairy/`` only.  The ``tests/`` tree is intentionally
  excluded so this very file (which discusses the pattern) does not
  self-trigger.
* **Comments / docstrings:** Lines whose entire content is a Python
  comment (starts with ``#`` ignoring leading whitespace) are
  skipped; this catches the post-fix comments throughout the
  library that document the migrated sentinel pattern.
* **Astype recovery exemption:** A match whose remainder-of-line
  contains ``.astype(`` is treated as the lower-severity P3
  "dtype-recovered" form and exempted.  The audit rated those
  benign; this pin focuses on the regression-grade P1 form.
* **Self-exemption:** Lines containing the literal string
  ``_PIN_PATTERN`` (the regex definition below) are skipped so this
  file itself does not match.

Canonical migration pattern (referenced in error messages):

    # OLD (deprecated, silent upcast hazard)
    obj = np.where(mask, E, 0.0 + 0.0j)

    # NEW (dtype-aware, preserves E.dtype)
    obj = np.where(mask, E, np.zeros((), dtype=E.dtype))
    #                            ^^^^^^^^^^^^^^^^^^^^^^^^

Or, for backend-agnostic code that may run on JAX / CuPy::

    xp = _xp_of(E)
    obj = xp.where(mask, E, xp.zeros((), dtype=E.dtype))

Author: Andrew Traverso -- v4.14.2 / Agent D
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest

import lumenairy as la


# ============================================================================
# Pattern definition
# ============================================================================

# Matches ``np.where(<anything>, 0.0 + 0.0j)`` allowing arbitrary
# whitespace around the ``+`` and the trailing literal.  The leading
# ``np.where(`` anchor restricts the match to actual call sites; this
# avoids false matches on the bare literal in scalar accumulators
# (``total = 0.0 + 0.0j``) or dict-default lookups
# (``coeffs.get(key, 0.0 + 0.0j)``) which the audit calls out as
# distinct, lower-risk patterns (the scalar literal carries an
# explicit complex128 promotion that's wanted in those contexts).
#
# Greedy ``.*`` (not ``[^)]*``) so the engine backtracks past nested
# parens in expressions like ``np.where(mask, np.exp(1j * phase),
# 0.0 + 0.0j)`` -- a more selective ``[^)]*`` would stop at the
# first ``)`` after ``phase`` and miss the anti-pattern entirely.
_PIN_PATTERN = re.compile(
    r'np\.where\(.*,\s*0\.0\s*\+\s*0\.0j\s*\)'
)


# Explicit allowlist of P3-rated sites the audit considers
# acceptable.  Each entry is ``(relative_posix_path, line_number,
# rationale)``.  The pin walker exempts these so the test reflects
# the audit's actual severity gradient (P1 fails, P3 documented).
# A future commit that wants to move a P3 site to P1-grade fix can
# remove the corresponding allowlist entry; the parametrized walker
# will then fail-loud at the offending site.
_P3_ALLOWLIST = {
    # ``ui/psf_mtf_dock.py:230`` -- audit P1-NEW-4 table row 4:
    # "UI code, lower priority".  The widget builds a one-shot
    # complex128 pupil at user-controlled refresh cadence; the
    # silent upcast is not on the performance-critical analysis
    # path.  v4.15+ Qt-side cleanup may migrate to the dtype-aware
    # sentinel; meanwhile, exempted.
    ('ui/psf_mtf_dock.py', 230),
}


def _is_pure_comment_line(line: str) -> bool:
    """Return True if the entire line is a Python comment.

    Catches lines like::

        # complex128 literal ``0.0 + 0.0j``.

    where the source-walk might otherwise match the literal even
    though it's just documentation.
    """
    stripped = line.lstrip()
    return stripped.startswith('#')


def _is_allowlisted(rel_path_posix: str, lineno: int) -> bool:
    """Return True if ``(rel_path_posix, lineno)`` is in the audit-
    documented P3 allowlist.  Tolerant of small line-number drift
    (+/- 5) since edits elsewhere in the file can shift the absolute
    line number without the actual anti-pattern site moving.
    """
    for allow_path, allow_ln in _P3_ALLOWLIST:
        if rel_path_posix.endswith(allow_path):
            if abs(lineno - allow_ln) <= 5:
                return True
    return False


def _walk_lumenairy_py_files() -> List[Path]:
    """Return every ``*.py`` file under ``lumenairy/`` (the installed
    library tree), excluding ``__pycache__``."""
    root = Path(la.__path__[0])
    return [
        p for p in root.rglob('*.py')
        if '__pycache__' not in p.parts
    ]


def _scan_file(path: Path) -> List[Tuple[int, str]]:
    """Return every ``(line_number, line_content)`` in ``path`` that
    matches the pin pattern AND is not exempted (pure comment,
    self-exempt, ``.astype(`` recovery, audit-documented P3
    allowlist).
    """
    hits: List[Tuple[int, str]] = []
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        # If we cannot read the file, surface that as a collection
        # error rather than silently skipping -- the source walk
        # depends on reading every .py file.
        raise
    # Compute the path relative to the lumenairy package root so
    # the allowlist keys are stable regardless of the absolute
    # install location.
    pkg_root = Path(la.__path__[0])
    try:
        rel_posix = path.relative_to(pkg_root).as_posix()
    except ValueError:
        rel_posix = path.as_posix()
    for i, line in enumerate(text.splitlines(), start=1):
        if _is_pure_comment_line(line):
            continue
        # Self-exempt: lines that reference the regex variable name
        # ``_PIN_PATTERN`` come from this very test file (or its
        # docstring imports / re-exports).  Skip them so the pin
        # does not match its own definition.
        if '_PIN_PATTERN' in line:
            continue
        match = _PIN_PATTERN.search(line)
        if match is None:
            continue
        # Exempt: if the remainder of the line (after the match end)
        # contains ``.astype(``, the dtype is recovered downstream;
        # the audit rates this lower-severity (P3) and exempts it
        # from the regression pin.
        remainder = line[match.end():]
        if '.astype(' in remainder:
            continue
        # Exempt: audit-documented P3 allowlist entries.
        if _is_allowlisted(rel_posix, i):
            continue
        hits.append((i, line.strip()))
    return hits


# Materialise the file list at module-import time so the test IDs
# show in pytest collection output (and so collection itself fails
# loudly if the walker breaks).
_LUMENAIRY_PY_FILES = sorted(_walk_lumenairy_py_files())


# ============================================================================
# Pre-flight -- the walker found at least one .py file
# ============================================================================

def test_walker_found_lumenairy_py_files():
    """Pre-flight: the file walker found at least one ``.py`` file in
    ``lumenairy/``.  Empty would mean the meta-pin below silently
    passes with zero parametrized cases -- a worse outcome than the
    P1-NEW-4 anti-pattern it is meant to catch.
    """
    assert len(_LUMENAIRY_PY_FILES) > 10, (
        f"File walker found only {len(_LUMENAIRY_PY_FILES)} .py files "
        f"under lumenairy/; expected many more.  The walker is "
        f"probably broken or ``la.__path__`` is wrong.")


# ============================================================================
# Pin -- no ``np.where(..., 0.0 + 0.0j)`` site without astype recovery
# ============================================================================

@pytest.mark.parametrize('py_path', _LUMENAIRY_PY_FILES,
                          ids=lambda p: str(p.relative_to(p.parents[1])))
def test_no_unguarded_zero_plus_zeroj_in_np_where(py_path):
    """For each ``.py`` file in ``lumenairy/``, no line may match
    ``np.where(..., 0.0 + 0.0j)`` without a trailing ``.astype(``
    recovery on the same line.

    Pre-fix (v4.14.1) the audit P1-NEW-4 caught 2 P1-severity sites:
    ``optimize/core.py:966`` (Agent B's scope) and
    ``analysis/phase_retrieval.py:402`` (Agent C's scope).  After
    those two fixes land in v4.14.2 this pin passes.  A future commit
    that re-introduces the bare ``0.0 + 0.0j`` form will trip the
    pin at the file:line, with a copy-paste-able migration suggestion
    in the failure message.
    """
    hits = _scan_file(py_path)
    if hits:
        # Build a helpful failure message with the canonical migration.
        rel = py_path.relative_to(py_path.parents[1])
        sites = '\n'.join(f'  {rel}:{ln}: {content}'
                          for ln, content in hits)
        pytest.fail(
            f"Found {len(hits)} unguarded ``np.where(..., 0.0 + 0.0j)`` "
            f"site(s) in {rel}:\n\n{sites}\n\n"
            "Migrate each to the dtype-aware sentinel:\n"
            "  obj = xp.where(mask, E, xp.zeros((), dtype=E.dtype))\n"
            "(where ``xp = _xp_of(E)`` for backend dispatch).\n"
            "Or, if local dtype recovery suffices, add a trailing "
            "``.astype(cdtype)`` on the same line.\n"
            "Audit reference: AUDIT_V4_14_1_2026_05_17.md P1-NEW-4."
        )


# ============================================================================
# Direct regression pins for the 2 P1 sites Agents B and C are closing
# ============================================================================

class TestP1New4SpecificSites:
    """Name-anchored regression pins for the 2 P1 sites the audit
    flagged.  Redundant with the parametrized walker above, but kept
    so a future regression of either named site is grep-able by the
    file:line pinned here.
    """

    def test_optimize_core_line_966_or_nearby_no_unguarded(self):
        """``optimize/core.py`` must contain NO unguarded
        ``np.where(..., 0.0 + 0.0j)`` site (Agent B's v4.14.2 fix
        target).  The exact line may shift across releases; what
        matters is that no match without ``.astype(`` recovery
        remains anywhere in the file.
        """
        opt_core = Path(la.__path__[0]) / 'optimize' / 'core.py'
        if not opt_core.exists():
            pytest.skip(f'{opt_core} not present in this checkout')
        hits = _scan_file(opt_core)
        assert hits == [], (
            f"optimize/core.py still has {len(hits)} unguarded "
            f"``np.where(..., 0.0 + 0.0j)`` site(s): {hits}.  "
            "This is the v4.14.1 audit P1-NEW-4 Agent B target.")

    def test_phase_retrieval_line_402_or_nearby_no_unguarded(self):
        """``analysis/phase_retrieval.py`` must contain NO unguarded
        ``np.where(..., 0.0 + 0.0j)`` site without ``.astype(``
        recovery on the same line (Agent C's v4.14.2 fix target).
        Lines 367, 548 are dtype-recovered via trailing
        ``.astype(cdtype)`` and exempted by the per-line filter.
        """
        pr_file = (Path(la.__path__[0]) / 'analysis'
                   / 'phase_retrieval.py')
        if not pr_file.exists():
            pytest.skip(f'{pr_file} not present in this checkout')
        hits = _scan_file(pr_file)
        assert hits == [], (
            f"analysis/phase_retrieval.py still has {len(hits)} "
            f"unguarded ``np.where(..., 0.0 + 0.0j)`` site(s) "
            f"without ``.astype(`` recovery: {hits}.  This is the "
            "v4.14.1 audit P1-NEW-4 Agent C target.")


# ============================================================================
# Counter-pin: the regex actually finds the anti-pattern when present
# ============================================================================

def test_pin_regex_matches_canonical_anti_pattern():
    """Sanity: confirm the regex matches the canonical anti-pattern in
    isolation.  Without this, a regex typo could silently make the
    pin a no-op.
    """
    sample = "obj = np.where(support, obj_new, 0.0 + 0.0j)"
    assert _PIN_PATTERN.search(sample) is not None, (
        f"Regex {_PIN_PATTERN.pattern!r} fails to match canonical "
        f"anti-pattern {sample!r}.")


def test_pin_regex_does_not_match_dtype_aware_form():
    """Counter-sanity: the dtype-aware sentinel form must NOT match
    the regex.  This is the migration target the failure messages
    point users toward.
    """
    sample = ("obj = np.where(support, obj_new, "
              "np.zeros((), dtype=obj_new.dtype))")
    assert _PIN_PATTERN.search(sample) is None, (
        f"Regex {_PIN_PATTERN.pattern!r} false-matched the dtype-"
        f"aware sentinel form {sample!r}; this would create false "
        "regressions on every correctly-migrated site.")


def test_pin_regex_does_not_match_scalar_accumulator():
    """Counter-sanity: bare scalar literals like
    ``total = 0.0 + 0.0j`` or ``out[key] = out.get(key, 0.0 + 0.0j)``
    must NOT match the regex.  Those are distinct, lower-risk usages
    that the audit calls out as acceptable (explicit complex128
    promotion is wanted in those contexts).
    """
    samples = [
        "total = 0.0 + 0.0j",
        "out[key] = out.get(key, 0.0 + 0.0j) + c",
        "E_pixel = 0.0 + 0.0j",
        "coeffs[key] = coeffs.get(key, 0.0 + 0.0j) + c",
    ]
    for s in samples:
        assert _PIN_PATTERN.search(s) is None, (
            f"Regex {_PIN_PATTERN.pattern!r} false-matched the "
            f"scalar accumulator pattern {s!r}; this would create "
            "false regressions on lines the audit considers benign.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
