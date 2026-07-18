"""v4.16.1 meta-pin (10th dispatcher walker): ``@lru_cache`` /
module-level ``_CACHE = OrderedDict(...)`` <-> ``_cache_registry``
enrollment walker.

Audit context
-------------

The v4.16.0 deep audit (``docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md``)
identified GLASS_VALIDITY <-> GLASS_REGISTRY and
``@lru_cache`` <-> ``_cache_registry`` enrollment as two sibling-gap
surfaces NOT covered by the 9 V-walkers landed in v4.14 -> v4.16.0.

This walker (the 10th dispatcher meta-pin) closes the second gap:
AST-walks every public ``lumenairy/*.py`` module for either:

1.  A ``@functools.lru_cache(...)`` or ``@lru_cache(...)`` decorator
    on a top-level function, OR
2.  A module-level cache declaration of the form
    ``_FOO_CACHE: 'OrderedDict[...]' = OrderedDict()`` (the canonical
    pattern used by every cache-owning module in the v4.14 -> v4.16.0
    series).

For each discovered cache, the walker verifies that the same module
contains at least one ``register_cache_clearer(...)`` call -- the
canonical enrollment site introduced in v4.16.0 / Agent D
(:mod:`lumenairy._cache_registry`).  A cache without enrollment
silently bypasses ``clear_asm_caches`` / ``clear_all_registered_caches``
-- the exact sibling-gap pattern the registry was supposed to retire.

Walker design
~~~~~~~~~~~~~

* AST-based discovery (cached via ``lru_cache(maxsize=None)`` per
  the v4.15.5 P3-NEW-F1-2 contract; ~30x speedup vs re-parsing each
  module on every test invocation).

* Discovery scope: ``lumenairy/**/*.py`` excluding
  ``_cache_registry.py`` itself (it IS the registry, not a client)
  and excluding ``__init__.py`` files (re-export shims do not host
  caches).

* Detection patterns:
  - ``@functools.lru_cache(...)`` / ``@lru_cache(...)`` on a top-
    level ``ast.FunctionDef`` whose name does not start with ``_``,
    OR matches one of the documented private-but-enrolled cache
    names (``_lg_polynomial_items``).
  - Module-level ``ast.AnnAssign`` / ``ast.Assign`` whose LHS name
    ends in ``_CACHE`` case-INSENSITIVELY (v5.17.1, audit P2-42: the
    historical case-sensitive filter let ``glass.py``'s lower-case
    ``_glass_cache`` / ``_glass_value_cache`` escape detection -- the
    exact sibling-gap this walker exists to retire) and whose RHS is
    ``OrderedDict(...)``, ``dict(...)``, or ``{}``.

* Enrollment verification: scan the module's AST for a top-level
  (or top-level-conditional, e.g. inside a try/except) ``ast.Call``
  whose ``.func`` resolves to ``register_cache_clearer`` (either
  bare-name or as ``X.register_cache_clearer`` attribute).

Exemption registry
~~~~~~~~~~~~~~~~~~

:data:`_CACHE_REGISTRY_EXEMPTIONS` lists ``(rel_path, cache_name)``
tuples for caches that legitimately do NOT need registry enrollment.
Each MUST cite WHY enrollment is not required.  Typical rationales:

* The cache is module-internal scaffolding (not user-facing state).
* The cache is the test-walker's own ``_file_to_ast`` lru_cache
  (test-internal; not part of the library's cache surface).
* The cache is a singleton built-once-per-process object (e.g.
  ``_JAX_IFT_SOLVER_CACHE`` -- a JAX-decorated solver, not a
  growable-keyed cache).

Counter-pin
~~~~~~~~~~~

:func:`test_counter_pin_fake_unregistered_cache_fails` injects a
synthetic cache declaration into the walker's discovered set with
NO sibling registration and asserts the main pin fires.  Modeled on
the v4.16.0 ``test_counter_pin_fake_missing_reexport_fails`` pattern.

Author: Andrew Traverso -- v4.16.1 / Agent C
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pytest

import lumenairy as la  # noqa: F401  -- import side-effects: register caches

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / 'lumenairy'


# ============================================================================
# Documented exemption registry.  Each entry is a (rel_posix_path,
# cache_name) tuple.  See the module docstring for the cited rationale.
# ============================================================================

_CACHE_REGISTRY_EXEMPTIONS = frozenset({
    # ---- _cache_registry.py itself ---------------------------------------
    # The registry's OWN _CACHE_CLEARERS dict is the registry; it isn't
    # registered.  We also skip the whole module via _iter_target_py_files,
    # but list the entry for documentation completeness.
    ('lumenairy/_cache_registry.py', '_CACHE_CLEARERS'),

    # ---- analysis/ao.py --------------------------------------------------
    # ``_DEFAULT_CACHE_CEILING_BYTES`` is a CONSTANT scalar (not a cache
    # container); the walker filters on assignment RHS shape but this
    # entry is named with the _CACHE suffix as a configuration knob.
    # Listed here for documentation completeness; the RHS-shape filter
    # already excludes it, so this entry is a no-op.
    ('lumenairy/analysis/ao.py', '_DEFAULT_CACHE_CEILING_BYTES'),

    # ---- propagators/asymptotic_jax_twin.py ------------------------------
    # ``_JAX_IFT_SOLVER_CACHE`` is a SINGLETON (built once per process,
    # never grows -- key is implicit "the solver").  v5.24.x (audit
    # S4-15): it IS now enrolled via ``register_cache_clearer(
    # 'jax_ift_solver', ...)`` + a ``clear_jax_ift_solver_cache`` helper,
    # because the singleton pins the compiled XLA executable's device
    # memory across ``clear_asm_caches`` -- draining it on an explicit
    # cache clear now reclaims that memory (the next solve rebuilds and
    # re-caches transparently).  This entry remains documentation-only:
    # the declaration is ``= None`` (RHS-shape-filtered by
    # ``_is_cache_rhs``) so the walker never discovers it, and the
    # module now carries a ``register_cache_clearer`` call regardless.
    ('lumenairy/propagators/asymptotic_jax_twin.py', '_JAX_IFT_SOLVER_CACHE'),
    # ``_HG_MODE_STACK_CACHE`` is cleared transitively by
    # ``clear_lg_mode_stack_cache`` (the LG/HG mode stack clearer drains
    # both LG and HG caches in one operation -- see ``clear_lg_mode_stack_cache``
    # in asymptotic_modes.py).  The registry enrolment is on ``lg_mode_stack``,
    # which under the hood drains BOTH.  v5.17.1 (audit P2-42): path
    # refreshed from the pre-v5.x ``propagators/asymptotic.py`` monolith;
    # asymptotic_modes.py also enrolls directly, so this entry is
    # documentation-only.
    ('lumenairy/propagators/asymptotic_modes.py', '_HG_MODE_STACK_CACHE'),

    # ---- propagators/propagation.py -------------------------------------
    # ``_FREQ_GRID_CACHE``, ``_BANDLIMIT_CACHE``, ``_H_CACHE``,
    # ``_PYFFTW_PLAN_CACHE``, ``_PYFFTW_BAD_SHAPES`` are all drained
    # transitively by ``_clear_local_asm_caches`` (registered as the
    # ``asm_local`` clearer in the central registry).  The registry
    # enrolment is a single ``asm_local`` entry; the per-cache
    # OrderedDict's are children, not separately enrolled siblings.
    ('lumenairy/propagators/propagation.py', '_FREQ_GRID_CACHE'),
    ('lumenairy/propagators/propagation.py', '_BANDLIMIT_CACHE'),
    ('lumenairy/propagators/propagation.py', '_H_CACHE'),
    ('lumenairy/propagators/propagation.py', '_PYFFTW_PLAN_CACHE'),
    ('lumenairy/propagators/propagation.py', '_PYFFTW_BAD_SHAPES'),

    # ---- analysis/phase_retrieval.py -------------------------------------
    # The kernels (GS / ER / HIO) are drained as a group by
    # ``clear_phase_retrieval_caches`` (the registered
    # ``phase_retrieval_kernels`` enrollment); each individual
    # OrderedDict isn't separately enrolled.
    ('lumenairy/analysis/phase_retrieval.py', '_ER_KERNEL_CACHE'),
    ('lumenairy/analysis/phase_retrieval.py', '_HIO_KERNEL_CACHE'),
})


# ============================================================================
# Walker primitives
# ============================================================================

@lru_cache(maxsize=None)
def _file_to_ast(path):
    """Parse ``path`` and return the module AST.  Cached per the
    v4.15.5 P3-NEW-F1-2 lru_cache contract for ~30x speedup vs
    re-parsing on every test invocation.
    """
    src = path.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(path))


def _iter_target_py_files():
    """Yield every public ``.py`` file in scope.

    Skips:

    * ``_cache_registry.py`` (IS the registry, not a client).
    * ``__init__.py`` files (re-export shims, not implementation files).
    * Files under any ``__pycache__`` directory.
    """
    for py in sorted(_PKG_ROOT.rglob('*.py')):
        if '__pycache__' in py.parts:
            continue
        if py.name == '__init__.py':
            continue
        if py.name == '_cache_registry.py':
            continue
        yield py


def _attr_terminal(node):
    """Return the terminal attribute name of an ``ast.Attribute`` or
    ``ast.Name``.  Returns None for other node types."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_lru_cache_decorator(deco):
    """True iff ``deco`` is an ``@lru_cache(...)``, ``@functools.lru_cache(...)``,
    or bare ``@lru_cache`` / ``@functools.lru_cache`` decorator."""
    # Strip the Call wrapper: ``@lru_cache(maxsize=...)`` is a Call whose
    # func is the Name/Attribute we care about.
    target = deco.func if isinstance(deco, ast.Call) else deco
    name = _attr_terminal(target)
    return name == 'lru_cache'


def _function_has_lru_cache_decorator(node):
    """True iff a ``FunctionDef`` carries any ``@lru_cache(...)`` /
    ``@functools.lru_cache(...)`` decorator."""
    for deco in node.decorator_list:
        if _is_lru_cache_decorator(deco):
            return True
    return False


def _is_cache_rhs(value_node):
    """True iff ``value_node`` is an ``OrderedDict(...)`` / ``dict(...)``
    / ``{}`` literal expression that the convention suggests is a
    cache container.

    The walker accepts:
    * ``OrderedDict()`` and ``OrderedDict(...)`` calls (any args)
    * ``dict()`` calls
    * ``{}`` Dict literal (with zero key/value pairs)

    The walker rejects:
    * Scalar / numeric literals (e.g. ``_DEFAULT_CACHE_CEILING_BYTES = 512 * 1024 * 1024``)
    * ``None`` (sentinel singletons handled below for completeness; the
      ``_JAX_IFT_SOLVER_CACHE = None`` pattern is listed as an exemption
      for the build-once singleton case).
    """
    if isinstance(value_node, ast.Call):
        terminal = _attr_terminal(value_node.func)
        return terminal in ('OrderedDict', 'dict')
    if isinstance(value_node, ast.Dict):
        # Empty-dict literal -- ``{}`` -- conventional cache initializer.
        return len(value_node.keys) == 0
    return False


def _discover_module_level_cache_assignments(tree):
    """Yield each top-level ``_<NAME>_CACHE = OrderedDict(...)``
    assignment in ``tree.body``.

    Returns a list of ``(cache_name, lineno)`` pairs.

    v5.17.1 (audit P2-42): the suffix match is case-INSENSITIVE
    (``_CACHE`` and ``_cache`` both qualify).  The historical
    case-sensitive ``endswith('_CACHE')`` let ``lumenairy/glass.py``'s
    lower-case ``_glass_cache`` / ``_glass_value_cache`` (which predate
    the ``_FOO_CACHE`` naming convention) bypass discovery entirely, so
    the main pin never evaluated them and their missing enrollment went
    unflagged for the module's whole life.
    """
    found = []
    for node in tree.body:
        # ``_FOO_CACHE: 'OrderedDict[...]' = OrderedDict()`` (AnnAssign)
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            if not target.id.upper().endswith('_CACHE'):
                continue
            if not _is_cache_rhs(node.value):
                continue
            found.append((target.id, node.lineno))
            continue
        # ``_FOO_CACHE = OrderedDict()`` (Assign)
        if isinstance(node, ast.Assign):
            if not _is_cache_rhs(node.value):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if not target.id.upper().endswith('_CACHE'):
                    continue
                found.append((target.id, node.lineno))
            continue
    return found


def _discover_lru_cache_functions(tree):
    """Yield each top-level function carrying an ``@lru_cache``
    decorator.  Returns a list of ``(func_name, lineno)`` pairs.
    """
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if _function_has_lru_cache_decorator(node):
            found.append((node.name, node.lineno))
    return found


def _module_has_register_cache_clearer_call(tree):
    """True iff the module's AST contains at least one Call node whose
    ``.func`` resolves to ``register_cache_clearer`` (bare-name OR as
    ``X.register_cache_clearer`` attribute), AND the call appears at
    **module level** (not nested inside a function / class / always-
    False ``if`` branch).

    v4.16.2 hardening (audit P2-NEW-F1-4): pre-v4.16.2 this walker
    walked the entire AST via ``ast.walk(tree)``, returning True on
    the first match.  A cache-owning module could satisfy the check
    with the call nested inside ``def some_test_only_function():``,
    an ``if False:`` branch, or an unreachable ``try/except`` -- the
    walker couldn't tell the difference.

    The hardened walker only accepts the call at module level,
    optionally inside a top-level ``try/except`` block (the
    canonical v4.16.0 cache-enrollment idiom) or a top-level ``if``
    block whose test is NOT a static-False (``False`` literal,
    ``0``, ``None``).  Calls nested any deeper -- inside function
    defs, class bodies, or always-False branches -- are rejected.
    """
    # Also accept the aliased name ``_register_cache_clearer`` which is
    # the canonical import-as in the v4.16.0 modules.
    canonical_names = {'register_cache_clearer', '_register_cache_clearer'}

    def _stmt_is_or_contains_module_level_call(stmt):
        """Recursively walk a module-level statement, but stop at
        function/class boundaries.  Returns True iff the call lives
        directly at module scope or inside a top-level
        try/except / if-not-always-False block."""
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            # Calls nested inside a def/class don't count.
            return False
        if isinstance(stmt, ast.If):
            # Reject always-False branches: ``if False:`` / ``if 0:``
            # / ``if None:``.  The body of the ``else`` (orelse) IS
            # acceptable because Python evaluates it instead.
            test = stmt.test
            if (isinstance(test, ast.Constant)
                    and not bool(test.value)):
                # Always-False -- skip the body, walk orelse only.
                for s in stmt.orelse:
                    if _stmt_is_or_contains_module_level_call(s):
                        return True
                return False
            # Otherwise walk both branches.
            for s in (*stmt.body, *stmt.orelse):
                if _stmt_is_or_contains_module_level_call(s):
                    return True
            return False
        if isinstance(stmt, ast.Try):
            # Top-level try is the canonical cache-enrollment idiom.
            for s in (*stmt.body, *stmt.orelse, *stmt.finalbody):
                if _stmt_is_or_contains_module_level_call(s):
                    return True
            # Skip handlers (the ImportError-pass branch is for
            # failure, not for the enrollment call).
            return False
        # Leaf-level: walk the statement's expression tree for a
        # qualifying Call.  This walks INTO Call args (so
        # `foo(register_cache_clearer(...))` would match), but
        # crucially does NOT walk INTO nested FunctionDef bodies
        # (those would have been caught above).
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                terminal = _attr_terminal(sub.func)
                if terminal in canonical_names:
                    return True
        return False

    for stmt in tree.body:
        if _stmt_is_or_contains_module_level_call(stmt):
            return True
    return False


# ============================================================================
# Discovery materialised at module-import time so collection failures
# are loud.
# ============================================================================

def _discover_all_caches() -> List[Tuple[str, str, int, str]]:
    """Walk every target module and yield discovery records.

    Returns a list of ``(rel_path, cache_name, lineno, kind)`` tuples
    where ``kind`` is one of ``'cache_dict'`` or ``'lru_cache'``.
    """
    records = []
    for py in _iter_target_py_files():
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        for name, lineno in _discover_module_level_cache_assignments(tree):
            records.append((rel, name, lineno, 'cache_dict'))
        for name, lineno in _discover_lru_cache_functions(tree):
            records.append((rel, name, lineno, 'lru_cache'))
    return sorted(records)


_DISCOVERED_CACHES = _discover_all_caches()


# ============================================================================
# Pre-flight: count floor.
# ============================================================================

def test_walker_minimum_cache_count():
    """Sanity floor: the walker MUST discover at least 10 caches
    across the library.

    The v4.16.0 baseline has at least 13 module-level cache
    declarations plus 1 ``@lru_cache``-decorated function.  Floor of
    10 leaves headroom for legitimate consolidation while catching a
    walker collapse (e.g. ``rglob`` returning nothing because the
    package was renamed).
    """
    count = len(_DISCOVERED_CACHES)
    assert count >= 10, (
        f"cache-registry enrollment walker found only {count} caches; "
        f"expected >= 10.  The discovery walker may have regressed."
    )


# ============================================================================
# Main pin: every discovered cache's owning module enrolls in the
# central registry, OR the cache is in _CACHE_REGISTRY_EXEMPTIONS.
# ============================================================================

def test_every_cache_owning_module_enrolls_with_registry():
    """Each module that owns a cache MUST contain at least one
    ``register_cache_clearer(...)`` call -- the canonical v4.16.0
    enrollment pattern.  Caches in :data:`_CACHE_REGISTRY_EXEMPTIONS`
    are skipped.

    A cache without enrollment silently bypasses
    ``clear_asm_caches`` / ``clear_all_registered_caches`` -- the
    sibling-gap pattern the central registry was meant to retire.
    The 10th meta-pin closes the gap by enforcing the contract
    structurally.
    """
    failures = []
    for rel, cache_name, lineno, kind in _DISCOVERED_CACHES:
        if (rel, cache_name) in _CACHE_REGISTRY_EXEMPTIONS:
            continue
        py = _REPO_ROOT / rel
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        if _module_has_register_cache_clearer_call(tree):
            continue
        failures.append(
            f"{rel}:{lineno} {cache_name} ({kind}): cache owner does "
            f"not call ``register_cache_clearer(...)`` anywhere in the "
            f"module.  Either:\n"
            f"    1. Add a module-level registration (mirroring the "
            f"v4.16.0 pattern in propagators/propagation.py:\n"
            f"           try:\n"
            f"               from .._cache_registry import register_cache_clearer as _register_cache_clearer\n"
            f"               import sys as _sys\n"
            f"               _this_mod = _sys.modules[__name__]\n"
            f"               _register_cache_clearer(\n"
            f"                   '<cache_name>',\n"
            f"                   lambda: getattr(_this_mod, '<clear_fn>')(),\n"
            f"               )\n"
            f"           except ImportError:\n"
            f"               pass\n"
            f"    OR\n"
            f"    2. Add ``({rel!r}, {cache_name!r})`` to "
            f"``_CACHE_REGISTRY_EXEMPTIONS`` in this file with a "
            f"cited rationale (e.g. 'singleton build-once-per-process', "
            f"'drained transitively by sibling cache_X clearer').")
    if failures:
        raise AssertionError(
            "Caches not enrolled in the central registry:\n  - "
            + "\n  - ".join(failures))


# ============================================================================
# Counter-pin: synthetic unregistered cache must trip the main pin.
# ============================================================================

def test_counter_pin_fake_unregistered_cache_fails(monkeypatch):
    """Inject a synthetic ``(rel_path, cache_name, lineno, kind)``
    record into the discovered set where the module does NOT call
    ``register_cache_clearer``, and assert the main pin fires.

    Without this counter-pin, a walker regression where the
    enrollment-detection short-circuits and accepts every module
    (e.g. ``_module_has_register_cache_clearer_call`` always returns
    True) would let the main pin pass on zero parametrized cases.

    The synthetic record points at the file ``lumenairy/_validation.py``,
    which holds no caches and no registry call; that file's name +
    a synthetic cache name simulate the failure mode.
    """
    fake_rel = 'lumenairy/_validation.py'
    fake_name = '_FAKE_UNREGISTERED_CACHE_FOR_COUNTERPIN'
    # Sanity: the fake (rel, name) is not exempt.
    assert (fake_rel, fake_name) not in _CACHE_REGISTRY_EXEMPTIONS
    # Sanity: the target file exists.
    assert (_REPO_ROOT / fake_rel).exists(), (
        "Counter-pin setup error: synthetic target file "
        "(lumenairy/_validation.py) does not exist."
    )
    # Sanity: the target file does NOT have a register_cache_clearer
    # call (otherwise the counter-pin would no-op because the module
    # passes the registration check).
    tree = _file_to_ast(_REPO_ROOT / fake_rel)
    assert not _module_has_register_cache_clearer_call(tree), (
        "Counter-pin setup error: synthetic target file "
        "(lumenairy/_validation.py) unexpectedly contains a "
        "``register_cache_clearer`` call -- pick a different sterile "
        "target file for the counter-pin."
    )

    fake_record = (fake_rel, fake_name, 1, 'cache_dict')

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_discovered = list(this_module._DISCOVERED_CACHES)
    monkeypatch.setattr(
        this_module,
        '_DISCOVERED_CACHES',
        real_discovered + [fake_record],
    )

    with pytest.raises(AssertionError) as excinfo:
        test_every_cache_owning_module_enrolls_with_registry()
    msg = str(excinfo.value)
    assert fake_name in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected cache. Got:\n{msg}"
    )
    assert fake_rel in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected file. Got:\n{msg}"
    )


def test_counter_pin_fake_registered_cache_does_not_trip(monkeypatch):
    """Sibling positive-signal counter-pin: a synthetic cache record
    in a file that DOES call ``register_cache_clearer`` must NOT trip.

    v5.1.0 (Agent C split): the canonical late-binding
    ``register_cache_clearer('asm_local', ...)`` call moved from
    ``propagators/propagation.py`` (now a thin re-export shell) to
    ``propagators/fft_infra.py``.  Use ``fft_infra.py`` as the
    registered-host file.
    """
    fake_rel = 'lumenairy/propagators/fft_infra.py'
    fake_name = '_FAKE_REGISTERED_CACHE_FOR_POSITIVE_COUNTERPIN'
    # Sanity: target has a registration call.
    tree = _file_to_ast(_REPO_ROOT / fake_rel)
    assert _module_has_register_cache_clearer_call(tree), (
        "Counter-pin setup error: positive-signal target file "
        "(lumenairy/propagators/fft_infra.py) lacks the expected "
        "``register_cache_clearer`` call."
    )
    fake_record = (fake_rel, fake_name, 1, 'cache_dict')

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_discovered = list(this_module._DISCOVERED_CACHES)
    monkeypatch.setattr(
        this_module,
        '_DISCOVERED_CACHES',
        real_discovered + [fake_record],
    )

    # Should NOT raise on the injected pair specifically; if the rest of
    # the codebase has real failures the AssertionError would still name
    # THOSE, but the synthetic name must not appear in the message.
    try:
        test_every_cache_owning_module_enrolls_with_registry()
    except AssertionError as exc:
        msg = str(exc)
        assert fake_name not in msg, (
            f"Positive-signal counter-pin: registered cache was "
            f"flagged. Got:\n{msg}"
        )


# ============================================================================
# Diagnostic helper: report the discovered cache inventory.
# ============================================================================

def test_discovered_cache_inventory_for_diagnostics():
    """Print discovered caches and their enrollment status for triage.

    Always passes -- the assertions live in
    :func:`test_every_cache_owning_module_enrolls_with_registry`.  This
    test exists so a developer triaging a meta-pin failure has a
    one-glance map of what the walker found.
    """
    items = []
    for rel, cache_name, lineno, kind in _DISCOVERED_CACHES:
        py = _REPO_ROOT / rel
        try:
            tree = _file_to_ast(py)
        except OSError:
            tag = 'PARSE_ERR'
        else:
            if (rel, cache_name) in _CACHE_REGISTRY_EXEMPTIONS:
                tag = 'exempt'
            elif _module_has_register_cache_clearer_call(tree):
                tag = 'enrolled'
            else:
                tag = 'MISSING'
        items.append((rel, cache_name, lineno, kind, tag))
    n_total = len(items)
    n_enrolled = sum(1 for *_, t in items if t == 'enrolled')
    n_exempt = sum(1 for *_, t in items if t == 'exempt')
    n_missing = sum(1 for *_, t in items if t == 'MISSING')
    print(
        f"\nv4.16.1 cache-registry walker discovered {n_total} caches:"
    )
    print(f"  [enrolled] {n_enrolled}")
    print(f"  [exempt]   {n_exempt}")
    print(f"  [MISSING]  {n_missing}")
    for rel, name, lineno, kind, tag in items:
        print(f"  [{tag:>8}] {rel}:{lineno}  {name}  ({kind})")
    assert isinstance(items, list)


# ============================================================================
# Direct contract pin: the v4.16.1 / Agent C late-binding lambda fix.
# ============================================================================

def test_propagation_asm_local_uses_late_binding_lambda():
    """``propagators/fft_infra.py`` MUST register ``_clear_local_asm_caches``
    via a late-binding lambda (the canonical v4.16.0 pattern used by
    the other 8 cache-owning modules).

    Pre-v4.16.1 the registration captured the function object directly
    (early-binding), which silently bypassed ``mock.patch.object`` --
    a tester-visible inconsistency vs the other 8 caches.  v4.16.1 /
    Agent C closes the gap per AUDIT_V4_16_0_DEEP P1-NEW-F1-2.

    v5.1.0 split (Agent C): the ``_clear_local_asm_caches`` function and
    its registration moved from ``propagators/propagation.py`` (now a
    thin re-export shell) to ``propagators/fft_infra.py``.  The test
    walks both candidate files so the contract continues to be pinned
    after the split.

    The test parses the AST and locates the
    ``_register_cache_clearer('asm_local', ...)`` Call node; the
    second argument must be a ``Lambda`` node (not a bare
    ``Name``-reference).
    """
    candidate_files = [
        _PKG_ROOT / 'propagators' / 'fft_infra.py',
        _PKG_ROOT / 'propagators' / 'propagation.py',
    ]
    found_call = None
    found_in = None
    for py in candidate_files:
        if not py.exists():
            continue
        tree = _file_to_ast(py)
        for sub in ast.walk(tree):
            if not isinstance(sub, ast.Call):
                continue
            terminal = _attr_terminal(sub.func)
            if terminal not in ('register_cache_clearer',
                                '_register_cache_clearer'):
                continue
            # Expect the first arg to be the string literal 'asm_local'.
            if not sub.args:
                continue
            first = sub.args[0]
            if (isinstance(first, ast.Constant)
                    and first.value == 'asm_local'):
                found_call = sub
                found_in = py
                break
        if found_call is not None:
            break
    assert found_call is not None, (
        "Could not locate the ``register_cache_clearer('asm_local', ...)`` "
        "call in propagators/fft_infra.py or propagators/propagation.py."
    )
    assert len(found_call.args) >= 2, (
        f"register_cache_clearer('asm_local', ...) call in {found_in} "
        f"must pass a second positional argument (the clear-function)."
    )
    second = found_call.args[1]
    assert isinstance(second, ast.Lambda), (
        f"v4.16.1 / Agent C (audit P1-NEW-F1-2) regression: "
        f"register_cache_clearer('asm_local', ...) in {found_in} second "
        f"argument is {type(second).__name__}, expected ast.Lambda "
        f"(late-binding closure mirroring the other 8 cache enrollments)."
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
