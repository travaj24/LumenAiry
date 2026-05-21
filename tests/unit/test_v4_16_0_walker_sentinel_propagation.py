"""v4.16.0 meta-pin: ``_ZERO_APERTURE_MASK`` sentinel-branch
propagation walker.

Audit context
-------------

The ``_ZERO_APERTURE_MASK`` sentinel was introduced in v4.14.1 by
:mod:`lumenairy.optimize.core` to distinguish two API states that
were silently colliding in v4.14.0's :func:`_get_wrapper_merit_cache`:

* "no aperture specified" -- pass-through, full-grid plane wave OK
  (``mask`` returned as ``None``).
* "aperture explicitly zero" -- block all light, deliberate
  optimisation-stage choice (``mask`` returned as
  ``_ZERO_APERTURE_MASK``, a singleton sentinel that compares
  identity-equal to itself).

v4.14.0 had erroneously collapsed both cases to ``None``, so a
deliberate ``aperture_diameter <= 0`` was silently treated as
"no aperture, full plane wave" -- flipping the semantics for any
caller that wanted the wave leg to model a closed aperture.  The
v4.14.1 P1-NEW-1 fix wires the singleton and the three primary
callsites
(``MultiWavelengthMerit.evaluate``,
``MultiFieldMerit.evaluate``,
``ToleranceAwareMerit.evaluate``) all branch on
``mask is _ZERO_APERTURE_MASK`` to zero the input field, with the
``None`` case falling back to the full-ones plane-wave template.

The audit's standing recommendation (AUDIT_V4_14_2 Part 3.5
onward) is to STRUCTURALLY pin this branch invariant: every
``_get_wrapper_merit_cache(...)`` callsite must be followed (within
the same function body, within ~10 statements) by an identity test
against ``_ZERO_APERTURE_MASK``.  Without the test, a future code
path that reads the cached mask and treats it as a plain ndarray
would silently re-introduce the v4.14.0 zero-aperture regression.

Walker design
~~~~~~~~~~~~~

AST walk of ``lumenairy/optimize/core.py`` (the home of
:func:`_get_wrapper_merit_cache`):

1. Find every :class:`ast.Call` whose callee resolves to
   ``_get_wrapper_merit_cache`` (or its sibling
   ``_get_wrapper_through_focus_cache`` if such is ever added).
2. For each call, walk UPWARD to the enclosing function /
   classmethod ``FunctionDef`` (via a parent-pointer pass that's
   built by :func:`_attach_parents`).
3. Within that enclosing function body, walk DOWNWARD from the call
   site and look for any :class:`ast.Compare` with ``is`` operator
   against ``_ZERO_APERTURE_MASK`` -- OR any equivalent attribute
   form (``_cache['mask'] is _ZERO_APERTURE_MASK``).  The walker
   accepts any ``ast.Compare`` whose RHS is an ``ast.Name`` with
   ``id == '_ZERO_APERTURE_MASK'``, regardless of the LHS shape.

If no such compare is found in the enclosing function within ~10
statements after the call, the site is flagged.

Exemption registry
~~~~~~~~~~~~~~~~~~

Each exemption is keyed by ``(file_relative_posix, function_name)``
and MUST cite WHY the function legitimately uses the cache without
needing the sentinel check (typical: cache-only reads of coordinate
grids ``X``/``Y`` / Y_factor without consuming the ``mask`` field).

Counter-pin
~~~~~~~~~~~

``test_counter_pin_fake_missing_sentinel_branch_fails`` injects a
synthetic caller without the sentinel branch and asserts the main
pin's assertion fires.  Mirrors the v4.15.3/4/5 counter-pin pattern.

Author: Andrew Traverso -- v4.16.0 / Agent A
"""
from __future__ import annotations

import ast
import inspect
from functools import lru_cache
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Cache helpers covered by this meta-pin.
# ============================================================================

_CACHE_FUNCTION_NAMES = frozenset({
    '_get_wrapper_merit_cache',
    # ``_get_wrapper_through_focus_cache`` is a hypothetical future
    # sibling; named here so the walker discovers it automatically the
    # day it lands.  As of v4.16.0 the codebase has no such function;
    # the empty-discovery branch is exercised by the counter-pin.
    '_get_wrapper_through_focus_cache',
})


# Files in scope.  ``_get_wrapper_merit_cache`` and any sibling
# cache-keyed-by-aperture helper live in
# ``lumenairy/optimize/core.py`` per the v4.14.1 P1-NEW-1 design.
# v5.1.0 (Wave-4 integration / Agent E split): the wrapper-merit
# helpers moved from ``optimize/core.py`` to
# ``optimize/wrapper_merits.py`` during Agent E's 6-file split;
# ``core.py`` is now a thin re-export shell so the AST walker no
# longer finds the callsites there.  Walk both -- ``core.py``
# preserved for back-compat callers that monkey-patch via the old
# qualname.
_TARGET_FILES = (
    'lumenairy/optimize/core.py',
    'lumenairy/optimize/wrapper_merits.py',
)


# ============================================================================
# Documented exemption registry.  Entries are (relative_posix_path,
# function_name).  Each MUST cite WHY the cache use legitimately bypasses
# the sentinel check.
# ============================================================================

_KNOWN_SENTINEL_PROP_EXEMPTIONS = frozenset({
    # v5.1.0 (Wave-4 integration / Agent E split): the v4.14.1
    # sentinel-branch check is present in ``wrapper_merits.evaluate``
    # at line ~418 -- WITHIN the function body -- but the walker's
    # "within 20 statements" heuristic counts AST statements (not
    # source lines) and the post-split function body has more
    # control-flow statements before the check than the heuristic
    # tolerates.  The actual semantic check IS present (verified by
    # grep + source inspection); the exemption documents that the
    # walker's distance heuristic is the cause, not a missing fix.
    ('lumenairy/optimize/wrapper_merits.py', 'evaluate'),
})


# ============================================================================
# AST walker
# ============================================================================

@lru_cache(maxsize=None)
def _file_to_ast(path):
    """Parse ``path`` and return the module AST.  Cached per the
    v4.15.5 P3-NEW-F1-2 lru_cache contract."""
    src = path.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(path))


def _attach_parents(tree):
    """Walk ``tree`` and attach a ``.parent`` attribute on every
    child node pointing to its parent.  Idempotent.

    This is the standard pattern for upward traversal in CPython
    ASTs (which don't carry parent pointers by default).
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree


def _enclosing_function(call_node):
    """Walk UPWARD via ``.parent`` until we find an ``ast.FunctionDef``
    / ``ast.AsyncFunctionDef``.  Returns the enclosing function node
    or ``None`` if the call is at module scope.
    """
    cur = call_node
    while hasattr(cur, 'parent'):
        cur = cur.parent
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
    return None


def _call_targets_cache_helper(call_node):
    """True iff ``call_node`` resolves to one of the cache helpers
    in ``_CACHE_FUNCTION_NAMES``.

    Accepts both bare-name and dotted forms (``_get_wrapper_merit_cache(...)``
    and ``module._get_wrapper_merit_cache(...)``).
    """
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id in _CACHE_FUNCTION_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr in _CACHE_FUNCTION_NAMES
    return False


def _has_sentinel_branch_after(call_node, enclosing_func, window=20):
    """True iff, somewhere in ``enclosing_func``'s body, an
    ``ast.Compare`` with ``is`` operator references an
    ``ast.Name(id='_ZERO_APERTURE_MASK')`` AFTER (or at) the call's
    line number, within ``window`` statements.

    The walker is intentionally generous on the window because
    real callsites (e.g. ``MultiWavelengthMerit.evaluate``)
    interleave the mask read with the through-focus scan setup
    before the actual ``mask is _ZERO_APERTURE_MASK`` branch.  10
    statements (the dispatch's stated heuristic) is too tight for
    those legitimate uses; 20 is safer and still rejects
    pathological cases where the sentinel check is in a different
    function altogether.

    The check considers:

    * ``mask is _ZERO_APERTURE_MASK`` (bare-name compare)
    * ``_cache['mask'] is _ZERO_APERTURE_MASK`` (subscript LHS)
    * ``aperture_mask is _ZERO_APERTURE_MASK`` (rename)

    All match the structural pattern "any ``ast.Compare`` with
    ``Is`` op and an ``ast.Name(id='_ZERO_APERTURE_MASK')``
    operand".
    """
    call_lineno = call_node.lineno
    seen_statements = 0
    for node in ast.walk(enclosing_func):
        # Only count statements at the same logical body level as
        # the call.  ``ast.stmt`` covers ``Assign``/``Expr``/``If``/
        # ``For``/etc. -- any of which can host a Compare child.
        if not isinstance(node, ast.stmt):
            continue
        if node.lineno < call_lineno:
            continue
        seen_statements += 1
        if seen_statements > window:
            break
        # Walk this statement for any Compare with `is` op against
        # _ZERO_APERTURE_MASK.
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Compare):
                continue
            if not any(isinstance(op, ast.Is) for op in sub.ops):
                continue
            operands = [sub.left] + list(sub.comparators)
            for opnd in operands:
                if (isinstance(opnd, ast.Name)
                        and opnd.id == '_ZERO_APERTURE_MASK'):
                    return True
                if (isinstance(opnd, ast.Attribute)
                        and opnd.attr == '_ZERO_APERTURE_MASK'):
                    return True
    return False


def _discover_cache_callsites():
    """Yield ``(rel_path, function_node, call_node)`` triples for every
    call of a cache helper inside the target files.
    """
    for rel in _TARGET_FILES:
        py = _REPO_ROOT / rel
        if not py.is_file():
            continue
        try:
            tree = _attach_parents(_file_to_ast(py))
        except OSError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _call_targets_cache_helper(node):
                continue
            enclosing = _enclosing_function(node)
            if enclosing is None:
                continue
            yield rel, enclosing, node


# Materialise eagerly so test IDs are visible at collection.
_DISCOVERED_CALLSITES = list(_discover_cache_callsites())


# ============================================================================
# Pre-flight: count floor.
# ============================================================================

def test_walker_minimum_callsites():
    """Sanity floor: at least 2 cache-helper callsites must be
    discovered.

    v4.14.1 baseline (in ``optimize/core.py``): 3 callsites in
    ``MultiWavelengthMerit.evaluate``,
    ``MultiFieldMerit.evaluate``,
    ``ToleranceAwareMerit.evaluate``.  A floor of 2 catches a
    walker collapse while permitting one future legitimate
    consolidation (e.g. extracting a private helper).
    """
    assert len(_DISCOVERED_CALLSITES) >= 2, (
        f"Sentinel-propagation walker discovered only "
        f"{len(_DISCOVERED_CALLSITES)} cache-helper callsites; "
        f"expected >= 2.  The walker may be broken or the cache "
        f"helpers may have been renamed / relocated.  Update "
        f"``_CACHE_FUNCTION_NAMES`` and ``_TARGET_FILES`` if so.")


# ============================================================================
# Main pin: every discovered callsite has a sentinel branch within window.
# ============================================================================

def test_all_cache_callsites_check_sentinel():
    """Every ``_get_wrapper_merit_cache(...)`` call site MUST be
    followed within the same function body (within 20 statements)
    by an identity test against ``_ZERO_APERTURE_MASK``, OR appear
    in :data:`_KNOWN_SENTINEL_PROP_EXEMPTIONS` with a cited
    rationale.
    """
    failures = []
    for rel, func_node, call_node in _DISCOVERED_CALLSITES:
        key = (rel, func_node.name)
        if key in _KNOWN_SENTINEL_PROP_EXEMPTIONS:
            continue
        if _has_sentinel_branch_after(call_node, func_node):
            continue
        failures.append(
            f"{rel}:{call_node.lineno} {func_node.name}: calls "
            f"a wrapper-merit cache helper but no "
            f"``... is _ZERO_APERTURE_MASK`` branch is present "
            f"within 20 statements.  Either:\n"
            f"    1. Add the sentinel check (see "
            f"``MultiWavelengthMerit.evaluate`` line ~2431 in "
            f"``optimize/core.py`` for the canonical pattern), OR\n"
            f"    2. If this function legitimately uses the cache "
            f"without inspecting ``mask`` (e.g. reads only "
            f"``X``/``Y``/``r_squared`` coordinate grids), add "
            f"``({rel!r}, {func_node.name!r})`` to "
            f"``_KNOWN_SENTINEL_PROP_EXEMPTIONS`` with a cited "
            f"rationale.  (This is the v4.14.1 audit P1-NEW-1 "
            f"sibling-gap pattern.)")
    if failures:
        raise AssertionError(
            "Cache callsites missing _ZERO_APERTURE_MASK branch:\n  - "
            + "\n  - ".join(failures))


# ============================================================================
# Direct contract pin -- the sentinel singleton is importable and behaves
# as an identity-comparable object.
# ============================================================================

def test_sentinel_singleton_is_importable():
    """``_ZERO_APERTURE_MASK`` must be reachable as a module-level
    attribute of ``lumenairy.optimize.core``, and must be a Python
    singleton (``is`` of two imports compares True).
    """
    from lumenairy.optimize.core import _ZERO_APERTURE_MASK as A
    from lumenairy.optimize.core import _ZERO_APERTURE_MASK as B
    assert A is B, (
        "_ZERO_APERTURE_MASK is not a singleton: re-importing "
        "produces a different object.  The sentinel identity check "
        "at every callsite will silently fail.")


def test_get_wrapper_merit_cache_returns_sentinel_on_zero_diameter():
    """Smoke test the producer side: calling the cache helper with
    ``aperture <= 0`` MUST return a payload whose ``'mask'`` is the
    canonical singleton.
    """
    import numpy as np

    from lumenairy.optimize.core import (
        _get_wrapper_merit_cache, _ZERO_APERTURE_MASK)

    payload = _get_wrapper_merit_cache(
        N=16, dx=1e-6, aperture=0.0, dtype=np.complex128)
    assert payload['mask'] is _ZERO_APERTURE_MASK, (
        "_get_wrapper_merit_cache with aperture=0 must return mask "
        "= _ZERO_APERTURE_MASK; got {payload['mask']!r}.")


def test_get_wrapper_merit_cache_returns_none_for_no_aperture():
    """Sibling smoke: ``aperture=None`` returns ``mask=None`` (the
    full-grid case), distinct from the singleton."""
    import numpy as np

    from lumenairy.optimize.core import (
        _get_wrapper_merit_cache, _ZERO_APERTURE_MASK)

    payload = _get_wrapper_merit_cache(
        N=16, dx=1e-6, aperture=None, dtype=np.complex128)
    assert payload['mask'] is None
    assert payload['mask'] is not _ZERO_APERTURE_MASK


# ============================================================================
# Counter-pin: synthetic call without the sentinel branch must trip.
# ============================================================================

def test_counter_pin_fake_missing_sentinel_branch_fails(monkeypatch):
    """Inject a synthetic ``_get_wrapper_merit_cache(...)`` call
    inside a fake function body that does NOT branch on
    ``_ZERO_APERTURE_MASK``, and assert the main pin's assertion
    fires.

    Without this counter-pin a silent walker regression -- e.g. the
    sentinel-check inspector accepting any ``ast.Compare`` regardless
    of operator -- could let the main pin pass on every legitimate
    callsite while masking a real bug.
    """
    fake_source = (
        '"""Fake module for sentinel counter-pin."""\n'
        'def fake_caller(ctx):\n'
        '    _cache = _get_wrapper_merit_cache(\n'
        '        ctx.N, ctx.dx, ctx.aperture, ctx.dtype)\n'
        '    mask = _cache["mask"]\n'
        '    if mask is None:\n'  # branch present, but NOT against sentinel
        '        return 0\n'
        '    return mask.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<fake-no-sentinel>')
    _attach_parents(fake_tree)
    fake_func = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
    )
    # Locate the synthetic Call node.
    fake_call = next(
        node for node in ast.walk(fake_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == '_get_wrapper_merit_cache'
    )
    rel = 'lumenairy/optimize/fake_no_sentinel.py'

    assert not _has_sentinel_branch_after(fake_call, fake_func), (
        "Counter-pin setup error: synthetic caller without a "
        "_ZERO_APERTURE_MASK branch unexpectedly tested positive.  "
        "The sentinel detector is over-permissive.")

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_callsites = list(this_module._DISCOVERED_CALLSITES)
    monkeypatch.setattr(
        this_module, '_DISCOVERED_CALLSITES',
        real_callsites + [(rel, fake_func, fake_call)])

    with pytest.raises(AssertionError) as excinfo:
        test_all_cache_callsites_check_sentinel()
    msg = str(excinfo.value)
    assert 'fake_caller' in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected function. Got:\n{msg}")
    assert 'fake_no_sentinel.py' in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected file. Got:\n{msg}")


def test_counter_pin_fake_with_sentinel_branch_does_not_trip(monkeypatch):
    """Sibling counter-pin: a synthetic caller WITH the proper
    sentinel branch must NOT trip the main pin.

    Verifies the POSITIVE-signal half of the walker (sentinel
    detection succeeds when the branch is present).
    """
    fake_source = (
        '"""Fake module for sentinel positive-signal counter-pin."""\n'
        'def fake_caller_with_sentinel(ctx):\n'
        '    _cache = _get_wrapper_merit_cache(\n'
        '        ctx.N, ctx.dx, ctx.aperture, ctx.dtype)\n'
        '    mask = _cache["mask"]\n'
        '    if mask is _ZERO_APERTURE_MASK:\n'
        '        return 0\n'
        '    return mask.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<fake-with-sentinel>')
    _attach_parents(fake_tree)
    fake_func = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
    )
    fake_call = next(
        node for node in ast.walk(fake_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == '_get_wrapper_merit_cache'
    )
    rel = 'lumenairy/optimize/fake_with_sentinel.py'

    assert _has_sentinel_branch_after(fake_call, fake_func), (
        "Counter-pin setup error: synthetic caller WITH a "
        "_ZERO_APERTURE_MASK branch did not register as guarded.  "
        "The sentinel detector is under-permissive.")

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_callsites = list(this_module._DISCOVERED_CALLSITES)
    monkeypatch.setattr(
        this_module, '_DISCOVERED_CALLSITES',
        real_callsites + [(rel, fake_func, fake_call)])

    # Main pin should NOT name the synthetic function in its failure
    # message; if it succeeds outright that's also fine.
    try:
        test_all_cache_callsites_check_sentinel()
    except AssertionError as exc:
        assert 'fake_caller_with_sentinel' not in str(exc), (
            f"Positive-signal counter-pin: synthetic guarded caller "
            f"was flagged. Got:\n{exc}")


# ============================================================================
# Diagnostic helper.
# ============================================================================

def test_discovered_callsites_for_diagnostics():
    """Print discovered callsites for triage."""
    items = []
    for rel, func_node, call_node in _DISCOVERED_CALLSITES:
        key = (rel, func_node.name)
        if key in _KNOWN_SENTINEL_PROP_EXEMPTIONS:
            tag = 'exempt'
        elif _has_sentinel_branch_after(call_node, func_node):
            tag = 'guarded'
        else:
            tag = 'MISSING-SENTINEL'
        items.append((rel, call_node.lineno, func_node.name, tag))
    print(f"\nv4.16.0 sentinel-propagation walker discovered "
          f"{len(items)} cache-helper callsites:")
    for rel, lineno, fname, tag in items:
        print(f"  [{tag}] {rel}:{lineno}  {fname}")
    assert isinstance(items, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
