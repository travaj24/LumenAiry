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

import ast
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
    # ``elements/doe.py:539`` -- ``T = np.where(inside, T, 0.0 + 0j)`` in
    # ``create_fresnel_zone_plate``.  Found 2026-09-12 (WP-A22) by the
    # structural walk below, which the per-line regex could not see because
    # the fill is spelled ``0j`` rather than ``0.0j``.
    #
    # P3 by MEASUREMENT, not by assumption: on this branch
    # ``T = np.exp(1j * np.where(is_even, 0.0, np.pi))`` and that phase is
    # built from PYTHON FLOAT literals, so it is float64 for every input the
    # entry point accepts -- ``T`` is complex128 unconditionally (measured
    # complex128 on all four (binary, n_zones) combinations) and the fill is
    # complex128 too.  Nothing is promoted on any reachable call.
    #
    # It is still the wrong spelling, and it would become P1 the moment the
    # phase is built at a narrower dtype.  The one-line migration is
    # ``np.where(inside, T, np.zeros((), T.dtype))``; recorded as a request to
    # that module's owner in WP-A22's report.  Remove this entry when it
    # lands -- the walk will then confirm it rather than exempt it.
    ('elements/doe.py', 539),
}


def _literal_or_none(node):
    """The value of a numeric literal expression, or ``None``.

    ``ast.literal_eval`` rather than a hand walk, because ``0.0 + 0.0j`` is a
    ``BinOp`` and not a ``Constant``: CPython's parser does not constant-fold
    (that happens later, on the bytecode), and ``literal_eval`` is the one
    place that understands the ``real + imagj`` form without executing
    anything.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return None


def _is_complex_zero_fill(node) -> bool:
    """True for a literal complex ZERO, in any spelling.

    ``0j``, ``0.0j`` and ``0.0 + 0.0j`` all qualify; they are the same defect,
    because a Python complex is WEAKLY typed in jnp and promotes the whole
    ``where`` to complex whenever the other branch is real.

    What does NOT qualify, and the distinction is the point: a NON-ZERO
    sentinel such as ``1.0 + 0.0j`` (``asymptotic.py``'s and the twin's
    ``safe_det``, ``_lens_traced.py``'s five direction-cosine guards).  Those
    exist to keep a subsequent division finite and their VALUE is load-bearing,
    so ``zeros((), dtype)`` is not the migration for them -- swapping one in
    would change the answer.  They are a different judgement call from the
    no-op zero fill this pin is named after, and a first draft of this walk
    that accepted any ``BinOp`` containing a complex zero flagged all six of
    them.  A real zero (``0.0``) does not qualify either: it promotes nothing.
    """
    value = _literal_or_none(node)
    return isinstance(value, complex) and value == 0


def _scan_file_ast(path: Path, text: str) -> List[Tuple[int, str]]:
    """The line regex above, done structurally -- and the reason it exists.

    ADDED 2026-09-12 (WP-A22).  The regex scan is per LINE, so it sees a
    ``where`` call only when the call and its fill are on the same line.  A
    black-formatted or hand-wrapped call hides from it completely::

        safe_phi = jnp.where(jnp.isfinite(jnp.abs(phi_star)), phi_star,
                             0.0 + 0.0j)

    That is not hypothetical: it is
    ``propagators/asymptotic_jax_twin.py``'s ``safe_phi``, which sat two lines
    below a site the pin DID catch and escaped it for the whole of v4.14-v5.45
    on nothing but where the line broke.  MEASURED before the fix: the regex
    reported 1 site in that file, this walk reports 2.

    Structural rather than textual, so a literal inside a comment or a string
    cannot match at all (no ``_is_pure_comment_line`` heuristic needed), and
    ``xp.where`` / ``jnp.where`` / a bare ``where(...)`` are all caught by the
    same rule rather than by the regex's incidental ``np.where`` substring hit.

    Returns ``(lineno, text)`` for the FILL's own line, so the failure message
    points at the literal rather than at the head of the call.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:                        # pragma: no cover
        raise AssertionError(
            f'{path}: cannot be parsed, so this pin cannot walk it: {exc}'
        ) from exc
    lines = text.splitlines()
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 3:
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name != 'where':
            continue
        fill = node.args[2]
        if not _is_complex_zero_fill(fill):
            continue
        # Exempt: the VALUE branch is itself a literal complex constant, so
        # the array is complex by construction and by intent and there is no
        # operand dtype for the fill to override.  ``doe.py:534``'s binary
        # zone mask, ``np.where(is_even & inside, 1.0 + 0j, 0.0 + 0j)``, is
        # the case: both branches are literals, the result is a complex
        # transmission mask, and nothing was promoted.  The defect this pin
        # is about needs a branch that CARRIES a dtype.
        if _literal_or_none(node.args[1]) is not None:
            continue
        # Same ``.astype(`` recovery exemption the regex scan applies, read
        # off the whole call rather than off one line: a wrapped call's
        # recovery can land on a different line from its fill.
        segment = '\n'.join(
            lines[node.lineno - 1:(node.end_lineno or node.lineno)])
        if '.astype(' in segment.split('where(', 1)[-1]:
            continue
        hits.append((fill.lineno, lines[fill.lineno - 1].strip()))
    return hits


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
    # The structural pass, for the calls the per-line regex cannot see (a
    # ``where`` whose fill sits on a continuation line).  Unioned rather than
    # substituted: the regex scan's exact behaviour is preserved, and this
    # only ever ADDS sites it was blind to.
    seen = {ln for ln, _ in hits}
    for ln, content in _scan_file_ast(path, text):
        if ln in seen or _is_allowlisted(rel_posix, ln):
            continue
        seen.add(ln)
        hits.append((ln, content))
    return sorted(hits)


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


# ============================================================================
# Falsifiability -- the structural walk catches what the line regex cannot
# ============================================================================

_TEETH_CASES = [
    # (label, source, regex-visible, must-be-flagged)
    ('same line, the spelling the pin is named after',
     'x = jnp.where(ok, v, 0.0 + 0.0j)\n', True, True),
    ('CONTINUATION LINE -- the whole reason for the AST walk',
     'y = jnp.where(jnp.isfinite(jnp.abs(phi)), phi,\n'
     '             0.0 + 0.0j)\n', False, True),
    ('non-zero sentinel, whose VALUE is load-bearing',
     'z = jnp.where(ok, det, 1.0 + 0.0j)\n', False, False),
    ('.astype recovery on the call',
     'w = jnp.where(ok, v, 0.0 + 0.0j).astype(v.dtype)\n', False, False),
    ('the migration this pin prescribes',
     'u = jnp.where(ok, v, jnp.zeros((), v.dtype))\n', False, False),
    ('both branches literal -- complex by construction',
     't = np.where(c, 1.0 + 0j, 0.0 + 0j)\n', False, False),
    ('a literal inside a comment is not code',
     '# np.where(ok, v, 0.0 + 0.0j)\n', False, False),
    ('a real zero fill promotes nothing',
     'q = jnp.where(ok, v, 0.0)\n', False, False),
]


@pytest.mark.parametrize(
    'label,source,regex_visible,flagged',
    _TEETH_CASES, ids=[c[0][:42] for c in _TEETH_CASES])
def test_the_structural_walk_has_teeth(label, source, regex_visible, flagged,
                                       tmp_path):
    """FAIL-BEFORE for the 2026-09-12 extension, on synthetic sources.

    The walker above runs over the real tree and reports zero, which is what
    a green gate looks like AND what a gate that has stopped working looks
    like.  These eight cases separate the two on this build.

    The second row is the one that matters: it is
    ``propagators/asymptotic_jax_twin.py``'s ``safe_phi`` verbatim, which
    escaped this pin from v4.14 to v5.45 purely because its literal fell on a
    continuation line.  MEASURED here: the regex sees 0, the walk sees 1.

    Rows 3-8 are the counter-pins.  A walk that flagged everything would pass
    row 2 and be useless; in particular a non-zero sentinel must NOT be
    flagged, because ``zeros((), dtype)`` is not its migration -- substituting
    one would change the answer, and a first draft of this walk did exactly
    that to six live sites before the rule was made value-correct.
    """
    path = tmp_path / 'sample.py'
    path.write_text(source, encoding='utf-8')

    regex_hits = []
    for i, line in enumerate(source.splitlines(), start=1):
        if _is_pure_comment_line(line):
            continue
        match = _PIN_PATTERN.search(line)
        if match is None or '.astype(' in line[match.end():]:
            continue
        regex_hits.append(i)
    assert bool(regex_hits) == regex_visible, (
        f'{label}: the per-line regex was expected to '
        f'{"see" if regex_visible else "miss"} this and did not.  The premise '
        f'of the row has changed, so it no longer tests what it claims.')

    ast_hits = _scan_file_ast(path, source)
    assert bool(ast_hits) == flagged, (
        f'{label}: the structural walk reported {len(ast_hits)} site(s), '
        f'expected {"at least one" if flagged else "none"}.  Source:\n'
        f'{source}')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
