"""v5.2 13th meta-pin walker -- shell-vs-canonical-location uniqueness (V13).

The v5.1.0 release split six monolithic 2200+ LOC files into ~35
topical submodules.  Each pre-split file is now a thin back-compat
re-export *shell*: its body should be limited to
``from .<submod> import <name>`` re-export lines (plus PEP-562
``__getattr__`` plumbing and module-level docstring / ``__all__``),
and the bodies of the re-exported callables / classes should live in
the submodules they were moved to.

The v5.1.0 audit (``docs/audits/AUDIT_V5_1_0_2026_05_20.md``)
identified P2-NEW-F2-1 finding #1: there is no general walker that
enforces this invariant.

> Shell-vs-canonical-location uniqueness pin -- no general walker.
> ``propagate_modal_asymptotic.__module__`` has a hand-written test
> for its specific monkey-patch contract, but no walker asserts "for
> every name in shell's ``from .X import Y`` block, ``Y.__module__``
> is the submodule."  A maintainer could reintroduce a function body
> to a shell module silently.

V13 closes that finding.  For each of the six shells:

    lumenairy/raytrace/core.py
    lumenairy/propagators/propagation.py
    lumenairy/propagators/asymptotic.py
    lumenairy/optimize/core.py
    lumenairy/io/prescriptions.py
    lumenairy/analysis/core.py

the walker:

1. Parses the shell's source AST and enumerates every
   ``from .<submod> import <name>`` (single-dot relative import of a
   single submodule).  ``from .<submod> import *`` is expanded via
   the submodule's runtime ``__all__``.  Absolute imports, dotted
   relative imports (``from ..X import Y``), and bare ``from .
   import X`` are skipped (they're not shell-vs-canonical claims).

2. Resolves each imported name on the runtime shell module and, if
   the attribute is a callable / class / staticmethod / function,
   asserts ``attr.__module__`` equals the canonical submodule
   path -- NOT the shell.  If ``attr.__module__`` ends up equal to
   the shell's own dotted name, the body lives in the shell, which
   is exactly the regression the audit asked us to pin against.

3. Data constants (values that are not callable / class / module)
   are skipped: re-exporting a sentinel / dtype / dict by value is
   fine; ``__module__`` carries no signal there.

Exemptions live in ``_V13_SHELL_EXEMPTIONS`` -- a tuple of (shell
dotted name, attribute name) pairs.  Each exemption MUST have an
inline comment citing the contract that requires the body to stay
in the shell.  The idiom is patterned after
``_LOCK_WITHOUT_CACHE_EXEMPTIONS`` in
``tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py``.

v5.2 (AUDIT_V5_1_0 P2-NEW-F2-1 #1 + AUDIT_V5_1_1 Part 6 P3 v5.2
candidates): the file-location convention follows V12
(``tests/unit/test_v5_2_walker_changelog_changeset.py``); the
file-header docstring style follows V11
(``tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py``).
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Shell registry -- the six v5.1.0 split shells.
# ---------------------------------------------------------------------------
#
# Each entry is (parametrize-id, dotted-module-name).  The dotted name
# is the canonical importable path; the source path is derived from it
# by ``_shell_source_path`` below.  Ordering matches the v5.1.0
# release-notes split-order so the parametrize report rows line up
# with the CHANGELOG bullet list.

_V13_SHELLS: Tuple[Tuple[str, str], ...] = (
    ('raytrace_core_shell',       'lumenairy.raytrace.core'),
    ('propagation_shell',         'lumenairy.propagators.propagation'),
    ('asymptotic_shell',          'lumenairy.propagators.asymptotic'),
    ('optimize_core_shell',       'lumenairy.optimize.core'),
    ('io_prescriptions_shell',    'lumenairy.io.prescriptions'),
    ('analysis_core_shell',       'lumenairy.analysis.core'),
)


# ---------------------------------------------------------------------------
# Exemption list -- (shell-dotted-name, attribute-name) pairs whose body
# legitimately lives in the shell.
# ---------------------------------------------------------------------------
#
# Each entry must be backed by a concrete contract that makes moving
# the body to its submodule a behaviour change.  The contract is
# documented inline.  Add new entries sparingly -- the default
# disposition for a "the body migrated back to the shell" diff is to
# move it back to the submodule, not to extend this exemption list.
#
# Idiom patterned after ``_LOCK_WITHOUT_CACHE_EXEMPTIONS`` in
# ``tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py``.

_V13_SHELL_EXEMPTIONS: Tuple[Tuple[str, str], ...] = (
    # --------------------------------------------------------------
    # asymptotic.py :: propagate_modal_asymptotic
    # --------------------------------------------------------------
    # The v4.14.1/Agent-A cold-start invariant pin
    # (``tests/unit/test_audit_fixes_v4_14_1_agent_a.py``) monkey-
    # patches ``_solve_envelope_stationary_batch`` on the shell
    # module (``lumenairy.propagators.asymptotic``) and expects
    # ``propagate_modal_asymptotic`` to honour the patch.  Python
    # resolves intra-module name lookups against the function's
    # ``__globals__`` (the module the function is *defined* in), so
    # moving ``propagate_modal_asymptotic`` to ``asymptotic_maslov``
    # would break the contract -- the spy installed on the shell
    # would no longer be observed.  Documented in the comment block
    # at ``lumenairy/propagators/asymptotic.py`` immediately above
    # the function definition (lines ~240-249).
    ('lumenairy.propagators.asymptotic', 'propagate_modal_asymptotic'),
)


_V13_SHELL_EXEMPTIONS_SET = frozenset(_V13_SHELL_EXEMPTIONS)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _shell_source_path(shell_dotted: str) -> Path:
    """Resolve ``lumenairy.<package>.<stem>`` to its on-disk ``.py``."""
    parts = shell_dotted.split('.')
    return _REPO_ROOT.joinpath(*parts).with_suffix('.py')


def _extract_relative_single_dot_imports(
        shell_dotted: str) -> List[Tuple[str, str, str]]:
    """Walk the shell's AST and return every relative-import-from-
    single-submodule citation.

    Returns
    -------
    list of (submod_dotted, imported_name, asname_or_imported_name)
        ``submod_dotted`` is the full canonical module path of the
        submodule (``lumenairy.<package>.<submod>``).  ``imported_name``
        is the name as it appears in the submodule (``Y`` in
        ``from .X import Y``).  ``asname_or_imported_name`` is the
        name as bound on the shell namespace (``Z`` in ``from .X
        import Y as Z`` or ``Y`` when there is no ``as`` clause).

    Skipped forms:

    * Absolute imports (``from lumenairy.foo import Y``) -- they don't
      claim a shell-vs-canonical pairing.
    * Dotted-name relative imports (``from ..foo import Y``,
      ``from .foo.bar import Y``) -- not the V13 split surface.
    * ``from . import X`` (no ``names`` from a named submodule, just
      a submodule name bound at top level) -- this binds a submodule
      object, not a callable / class, and ``X.__module__`` would
      trivially be the submodule itself.
    * Star imports are EXPANDED via the submodule's runtime ``__all__``
      below (see ``_expand_star_imports``).
    """
    src_path = _shell_source_path(shell_dotted)
    assert src_path.is_file(), (
        f'V13 shell source not found at {src_path}; the parametrize '
        f'registry _V13_SHELLS is stale or the shell file was renamed.')
    src = src_path.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(src_path))

    shell_pkg = '.'.join(shell_dotted.split('.')[:-1])

    out: List[Tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # level == 1 == single leading dot.  level == 0 would be
        # absolute; level >= 2 means ``from ..`` or deeper.  We only
        # care about ``from .X import Y`` -- same-package relative.
        if node.level != 1:
            continue
        # ``from . import X`` has module == None -- skip (not a
        # claim about a re-exported callable's canonical module).
        if node.module is None:
            continue
        # Dotted relative module (``from .X.Y import Z``) -- not the
        # V13 surface either.  The split moved bodies to direct
        # sibling submodules, not nested packages.
        if '.' in node.module:
            continue
        submod_dotted = f'{shell_pkg}.{node.module}'
        for alias in node.names:
            imported_name = alias.name
            bound_name = alias.asname or imported_name
            out.append((submod_dotted, imported_name, bound_name))
    return out


def _expand_star_imports(
        shell_dotted: str,
        raw_imports: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Replace each ``(submod, '*', '*')`` triple with the expanded list
    of ``(submod, name, name)`` triples from the submodule's runtime
    ``__all__``.

    Submodules without an ``__all__`` fall back to ``dir(submod)``
    filtered to non-underscore names (matching Python's default
    star-import semantics).
    """
    expanded: List[Tuple[str, str, str]] = []
    for submod_dotted, imported, bound in raw_imports:
        if imported != '*':
            expanded.append((submod_dotted, imported, bound))
            continue
        submod = importlib.import_module(submod_dotted)
        all_names = getattr(submod, '__all__', None)
        if all_names is None:
            all_names = [n for n in dir(submod) if not n.startswith('_')]
        for name in all_names:
            expanded.append((submod_dotted, name, name))
    return expanded


# ---------------------------------------------------------------------------
# Runtime check -- ``obj.__module__`` must point to the submodule.
# ---------------------------------------------------------------------------


def _attr_has_module_signal(obj) -> bool:
    """True iff ``obj.__module__`` carries shell-vs-canonical signal.

    Functions, classes, methods, and other callables expose a
    ``__module__`` attribute that names the module the body was
    *defined* in.  Re-binding the name in another module does NOT
    change ``__module__``.  That's the load-bearing property V13
    relies on.

    Data constants (ints, strings, tuples, sentinel instances, dtype
    objects, numpy ufuncs that aren't real Python functions) either
    have no ``__module__`` or have one that doesn't reflect a
    Python-source definition location.  Skip those: re-exporting them
    by value is fine.

    Module objects also have ``__module__`` semantics that don't
    apply (a submodule re-exported via ``from .X import X`` has
    ``__module__`` == its own name, which would falsely look "in the
    shell" or "in the submodule" depending on direction).  Skip
    modules too.
    """
    if inspect.ismodule(obj):
        return False
    # ``inspect.isfunction`` covers plain ``def`` and ``async def``.
    # ``inspect.isclass`` covers ``class``.  ``inspect.isbuiltin``
    # covers C-level builtins (numpy ufuncs etc.) -- their
    # ``__module__`` points at their defining C extension, which is
    # never a lumenairy shell, so the check would be a no-op anyway.
    # ``inspect.ismethod`` covers bound methods (rare at module
    # scope but possible).  We also accept any object that has a
    # writable ``__module__`` *and* is callable, to cover decorated
    # functions, ``functools.partial``-wrapped helpers, and
    # ``staticmethod`` / ``classmethod`` descriptors that survive at
    # module level.
    if inspect.isfunction(obj) or inspect.isclass(obj):
        return True
    if inspect.ismethod(obj) or inspect.isbuiltin(obj):
        # Builtins live in C extensions; their ``__module__`` will
        # never be a lumenairy shell.  Returning True here is safe
        # but adds noise; gate on a Python-source ``__module__``.
        mod_attr = getattr(obj, '__module__', None)
        return (isinstance(mod_attr, str)
                and mod_attr.startswith('lumenairy.'))
    if callable(obj):
        mod_attr = getattr(obj, '__module__', None)
        return (isinstance(mod_attr, str)
                and mod_attr.startswith('lumenairy.'))
    return False


def _check_one_shell(shell_dotted: str) -> List[str]:
    """Return a list of human-readable failure strings for the shell.

    An empty list means the shell is clean.  Each failure string
    cites the shell, the attribute name, the observed module, and
    the expected (submodule) module.
    """
    raw = _extract_relative_single_dot_imports(shell_dotted)
    expanded = _expand_star_imports(shell_dotted, raw)

    shell_mod = importlib.import_module(shell_dotted)

    failures: List[str] = []
    # De-dup on (bound name) -- a re-export listed in BOTH a star
    # import and an explicit ``from .X import Y as Z`` line should
    # be checked once.  We key on the bound name for the
    # de-duplication; submod-of-record is the first-encountered.
    seen_bound: dict = {}
    for submod_dotted, imported_name, bound_name in expanded:
        seen_bound.setdefault(bound_name, (submod_dotted, imported_name))

    for bound_name, (submod_dotted, imported_name) in sorted(
            seen_bound.items()):
        # Skip ``__all__`` re-imports and dunder names -- they're
        # not callables.
        if bound_name.startswith('__') and bound_name.endswith('__'):
            continue
        # The shell might rename or omit names; if the bound name
        # isn't actually on the shell namespace, the shell never
        # exposed it and there's nothing to check.
        if not hasattr(shell_mod, bound_name):
            continue
        obj = getattr(shell_mod, bound_name)
        if not _attr_has_module_signal(obj):
            continue
        obj_module = getattr(obj, '__module__', None)
        if not isinstance(obj_module, str):
            continue
        # The contract: the body should live in the submodule, not
        # the shell.  ``obj.__module__`` == shell-dotted-name means
        # the body is in the shell.
        if obj_module != shell_dotted:
            continue
        # Body lives in the shell.  Check the exemption list.
        if (shell_dotted, bound_name) in _V13_SHELL_EXEMPTIONS_SET:
            continue
        failures.append(
            f'{shell_dotted}.{bound_name}: body lives in the shell '
            f'(__module__ == {obj_module!r}); expected canonical '
            f'submodule {submod_dotted!r} per ``from .{submod_dotted.rsplit(".", 1)[-1]} '
            f'import {imported_name}'
            + (f' as {bound_name}'
               if bound_name != imported_name else '')
            + '`` in the shell source.  Move the function / class '
            f'body to {submod_dotted!r} and keep only the re-export '
            f'line in the shell -- OR, if this name has a contract '
            f'that requires the body to stay in the shell (e.g. a '
            f'monkey-patch test patches a sibling name on the shell '
            f'module), add ({shell_dotted!r}, {bound_name!r}) to '
            f'``_V13_SHELL_EXEMPTIONS`` in '
            f'``{Path(__file__).name}`` with an inline comment '
            f'citing the contract.')
    return failures


# ---------------------------------------------------------------------------
# Pre-flight -- the walker MUST discover at least one re-export per shell.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shell_id, shell_dotted', _V13_SHELLS,
                         ids=[s[0] for s in _V13_SHELLS])
def test_walker_found_relative_imports_in_shell(shell_id, shell_dotted):
    """Each shell must contain at least one ``from .<submod> import
    <name>`` line.  Zero would mean either the file isn't a shell at
    all (regression -- the split was undone) OR the AST walker is
    broken.  Either is a hard failure of the V13 invariant.
    """
    raw = _extract_relative_single_dot_imports(shell_dotted)
    assert raw, (
        f'V13 walker found ZERO ``from .<submod> import ...`` lines '
        f'in {shell_dotted}.  The file should be a thin re-export '
        f'shell per the v5.1.0 split; if the split was undone the '
        f'walker collapses to zero checks and the invariant silently '
        f'no-ops -- exactly the failure mode V13 was built to catch.')


# ---------------------------------------------------------------------------
# V13.main -- shell-vs-canonical-location uniqueness check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shell_id, shell_dotted', _V13_SHELLS,
                         ids=[s[0] for s in _V13_SHELLS])
def test_shell_reexports_have_submodule_canonical_module(
        shell_id, shell_dotted):
    """For every name re-exported from the shell via ``from .<submod>
    import <name>``, the runtime object's ``__module__`` must be the
    submodule -- not the shell.

    A function / class whose ``__module__`` equals the shell's own
    dotted name has its body in the shell, which is precisely the
    regression P2-NEW-F2-1 #1 flagged: a maintainer could
    silently reintroduce a body to a shell module and the v5.1.0
    split invariant would degrade without anyone noticing.

    Exemptions live in ``_V13_SHELL_EXEMPTIONS`` -- documented
    monkey-patch / globals-resolution contracts that require the
    body to stay in the shell.  Each exemption must be inline-
    commented with the contract that justifies it.
    """
    failures = _check_one_shell(shell_dotted)
    assert not failures, (
        'V13 shell-vs-canonical-location uniqueness violation(s) in '
        f'{shell_dotted}:\n  - '
        + '\n  - '.join(failures)
        + '\n\nClosure of audit AUDIT_V5_1_0_2026_05_20.md '
          'P2-NEW-F2-1 finding #1.'
    )


# ---------------------------------------------------------------------------
# V13.exempt -- the exemption list must stay grounded.
# ---------------------------------------------------------------------------


def test_exemption_list_entries_are_live():
    """Every (shell, attr) entry in ``_V13_SHELL_EXEMPTIONS`` must
    correspond to an attribute that (a) exists on the shell, (b) has
    its body in the shell (``__module__`` == shell-dotted-name).

    A stale exemption is itself a sibling-gap meta-pattern: an entry
    that no longer applies (the contract was retired, the function
    was moved out of the shell, the attribute was renamed) gives
    future readers a false impression that the shell legitimately
    hosts the body.  This pin keeps the exemption list pinned to
    reality.
    """
    stale: List[str] = []
    for shell_dotted, attr_name in _V13_SHELL_EXEMPTIONS:
        try:
            shell_mod = importlib.import_module(shell_dotted)
        except ImportError as exc:  # pragma: no cover -- guards typos
            stale.append(
                f'({shell_dotted!r}, {attr_name!r}): shell module is '
                f'not importable ({exc!r}).  Remove the exemption '
                f'entry or fix the dotted name.')
            continue
        if not hasattr(shell_mod, attr_name):
            stale.append(
                f'({shell_dotted!r}, {attr_name!r}): attribute is '
                f'not present on the shell.  Remove the exemption '
                f'entry -- the function / class no longer exists at '
                f'this location.')
            continue
        obj = getattr(shell_mod, attr_name)
        obj_module = getattr(obj, '__module__', None)
        if obj_module != shell_dotted:
            stale.append(
                f'({shell_dotted!r}, {attr_name!r}): attribute '
                f'__module__ is {obj_module!r}, not the shell.  The '
                f'body has been migrated to its canonical submodule '
                f'-- the V13 walker would now pass this name without '
                f'the exemption.  Remove the entry from '
                f'``_V13_SHELL_EXEMPTIONS`` to keep the list pinned '
                f'to reality.')
    assert not stale, (
        'V13 exemption list has stale entries:\n  - '
        + '\n  - '.join(stale))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
