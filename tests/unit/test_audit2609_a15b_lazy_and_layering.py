"""WP-A15b -- PEP 562 lazy solver loading and the import-time layering fix.

Findings of ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`` section
14 (TESTS-ARCH):

* **P2-7** -- ``import lumenairy`` eagerly pulled the rigorous-solver stack
  (``berreman`` -> ``rcwa`` -> ``backend.scipy``) for a caller who only wanted
  ``propagate_asm``.  ``lumenairy/elements/__init__.py`` and the solver blocks
  of ``lumenairy/__init__.py`` now resolve those names through a module
  ``__getattr__`` (PEP 562).
* **P2-8** -- 12 layering violations, ELEVEN of them already lazy and one at
  module level: ``raytrace/surface.py -> elements.lenses``.  That import moved
  into the single function that uses it.

What the lazy change must NOT break is the interesting half, and it is what
most of this file pins: ``from lumenairy.elements import rcwa``,
``import lumenairy.elements.rcwa as r``, ``lumenairy.elements.rcwa.X``,
``pickle`` round-trips of objects defined in those modules, ``dir()``,
``from ... import *``, every name in each ``__all__``, and ``AttributeError``
(never ``ImportError``) for an unknown name.

Oracle
------
The "is it still eager?" question is answered STATICALLY, by the same
measurement the audit made: an AST walk that separates module-level imports
(executed at import time) from in-function ones, then the transitive closure
of the module-level edges out of ``lumenairy/__init__.py``.  Static rather
than a subprocess probe deliberately -- the closure is what the property IS,
it cannot be perturbed by whatever a sibling test happened to import first,
and it needs no interpreter spawn.  ``if TYPE_CHECKING:`` bodies are excluded
because they do not execute at import time (``algebra/base.py:42``,
``raytrace/bundles.py:43-44`` and ``propagators/system.py:26-27`` are exactly
that and are NOT import-time edges).
"""
from __future__ import annotations

import ast
import importlib
import os
import pickle
import sys

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements as ELEM

_LAZY_SOLVER_SUBMODULES = ('berreman', 'bor', 'eme', 'pmm', 'rcwa')


# ===========================================================================
# The module-level import graph (shared oracle)
# ===========================================================================

def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING':
        return True
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def _module_level_statements(tree: ast.Module):
    """Statements that really execute at import time.

    Descends into module-scope ``if`` / ``try`` (both execute) but NOT into
    ``if TYPE_CHECKING:`` (never executes) and not into function or class
    bodies (executed at call time, which is the whole point of the
    in-function-import idiom this library uses for 11 of its 12 layering
    edges).
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If):
            if _is_type_checking(node):
                continue
            stack.extend(node.body)
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Try):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
            for handler in node.handlers:
                stack.extend(handler.body)
            continue
        yield node


def _resolve(anchor_package: str, node) -> str:
    """The absolute module a module-level import statement names.

    ``anchor_package`` is the package a relative import is resolved against:
    for ``pkg/__init__.py`` that is ``pkg`` ITSELF (``from .x import y``
    inside ``lumenairy/__init__.py`` means ``lumenairy.x``), for
    ``pkg/mod.py`` it is ``pkg``.  Getting this wrong makes every facade's
    edges resolve to a package that does not exist, which is silent and
    makes an import-graph assertion pass vacuously.
    """
    if isinstance(node, ast.Import):
        return node.names[0].name
    base = node.module or ''
    if not node.level:
        return base
    parts = anchor_package.split('.')
    up = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
    return '.'.join(up + ([base] if base else []))


def _module_level_edges():
    """``{importer: {(imported, lineno), ...}}`` over the whole package.

    ``lumenairy/ui/`` is excluded: it is the Qt application layer, is not
    imported by ``import lumenairy``, and the audit scopes its census the
    same way.
    """
    root = os.path.dirname(la.__file__)
    parent = os.path.dirname(root)
    edges: dict = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if '__pycache__' in dirpath or f'{os.sep}ui' in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, parent)
            mod = rel[:-3].replace(os.sep, '.')
            if mod.endswith('.__init__'):
                mod = mod[:-len('.__init__')]
                anchor = mod                      # a package: anchored on itself
            else:
                anchor = mod.rsplit('.', 1)[0]    # a module: on its package
            with open(path, encoding='utf-8', errors='replace') as fh:
                src = fh.read()
            try:
                tree = ast.parse(src)
            except SyntaxError:                   # pragma: no cover - env
                continue
            out = edges.setdefault(mod, set())
            for node in _module_level_statements(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                target = _resolve(anchor, node)
                if target.startswith('lumenairy'):
                    out.add((target, node.lineno))
    return edges


def _import_time_closure(entry='lumenairy'):
    """Every module a bare ``import <entry>`` executes, transitively.

    Importing ``lumenairy.elements.rcwa`` also executes ``lumenairy.elements``
    and ``lumenairy``, so each target contributes its own ancestors.
    """
    edges = _module_level_edges()
    seen, stack = set(), [entry]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        parts = mod.split('.')
        for i in range(1, len(parts)):
            anc = '.'.join(parts[:i])
            if anc.startswith('lumenairy') and anc not in seen:
                stack.append(anc)
        for target, _lineno in edges.get(mod, ()):
            if target not in seen:
                stack.append(target)
    return seen


# ===========================================================================
# P2-7 -- the five solver subpackages are not imported by ``import lumenairy``
# ===========================================================================

def test_bare_import_lumenairy_executes_no_rigorous_solver_module():
    """FAIL-BEFORE: on the pre-fix tree the closure contains 24 solver
    modules -- ``elements.berreman``, ``elements.bor`` (+8 of its own),
    ``elements.pmm`` (+8) and ``elements.rcwa`` (+4) -- because
    ``lumenairy/__init__.py`` and ``elements/__init__.py`` imported names
    from all four at module level.
    """
    closure = _import_time_closure('lumenairy')
    loaded = sorted(
        m for m in closure
        if m.startswith('lumenairy.elements.')
        and m.split('.')[2] in _LAZY_SOLVER_SUBMODULES)
    assert loaded == [], (
        f'``import lumenairy`` still executes the rigorous-solver stack: '
        f'{loaded}.  The lazy tables in lumenairy/__init__.py '
        f'(_LAZY_SOLVER_NAMES) and lumenairy/elements/__init__.py '
        f'(_LAZY_NAMES / _LAZY_SUBMODULES) are what keep this empty.')


def test_neither_facade_imports_a_solver_subpackage_at_module_level():
    """The two files the lazy tables live in, named explicitly, so a
    re-hoisted ``from .elements.rcwa import ...`` is reported at its line."""
    edges = _module_level_edges()
    offenders = []
    for facade in ('lumenairy', 'lumenairy.elements'):
        for target, lineno in sorted(edges.get(facade, ())):
            parts = target.split('.')
            if (len(parts) > 2 and parts[1] == 'elements'
                    and parts[2] in _LAZY_SOLVER_SUBMODULES):
                offenders.append(f'{facade}:{lineno} -> {target}')
    assert not offenders, (
        'eager solver import re-introduced in a facade:\n  - '
        + '\n  - '.join(offenders))


@pytest.mark.parametrize('sub', _LAZY_SOLVER_SUBMODULES)
def test_submodule_reachable_by_every_import_spelling(sub):
    """The ways a caller reaches a lazy subpackage must all work and must all
    return the SAME module object."""
    by_getattr = getattr(ELEM, sub)
    by_import = importlib.import_module(f'lumenairy.elements.{sub}')
    assert by_getattr is by_import
    assert sys.modules[f'lumenairy.elements.{sub}'] is by_getattr
    # ``import lumenairy.elements.rcwa as r`` binds the parent attribute, so
    # after the import the name is a plain attribute, not a __getattr__ hop.
    assert vars(ELEM)[sub] is by_import


def test_elements_dir_lists_the_lazy_names():
    d = set(dir(ELEM))
    for sub in _LAZY_SOLVER_SUBMODULES:
        assert sub in d, f'dir(lumenairy.elements) lost {sub!r}'
    for name in ('rcwa_efficiency_1d', 'pmm_efficiency_2d', 'BerremanStack',
                 'RCWAStack', 'PMMStack'):
        assert name in d, f'dir(lumenairy.elements) lost {name!r}'


def test_top_level_dir_lists_the_lazy_names():
    d = set(dir(la))
    for name in ('RCWAStack', 'PMMStack', 'BORStack', 'BerremanStack',
                 'pmm_2d_order_drift', 'rcwa_efficiency_1d'):
        assert name in d, f'dir(lumenairy) lost {name!r}'


def test_every_elements_all_entry_resolves():
    missing = [n for n in ELEM.__all__ if not hasattr(ELEM, n)]
    assert not missing, (
        f'lumenairy.elements.__all__ lists names that no longer resolve: '
        f'{missing}')


def test_every_top_level_all_entry_resolves():
    missing = [n for n in la.__all__ if not hasattr(la, n)]
    assert not missing, (
        f'lumenairy.__all__ lists names that no longer resolve: {missing}')


@pytest.mark.parametrize('mod', ['lumenairy', 'lumenairy.elements'])
def test_unknown_attribute_raises_attribute_error_not_import_error(mod):
    """``hasattr`` and ``getattr(..., default)`` must keep working, which
    they only do if ``__getattr__`` raises ``AttributeError``."""
    m = importlib.import_module(mod)
    with pytest.raises(AttributeError) as exc:
        getattr(m, 'no_such_name_a15b')
    assert 'no_such_name_a15b' in str(exc.value)
    assert not isinstance(exc.value, ImportError)
    assert hasattr(m, 'no_such_name_a15b') is False
    assert getattr(m, 'no_such_name_a15b', 'dflt') == 'dflt'


def test_lazy_names_resolve_to_the_same_object_as_a_direct_import():
    """Identity, not equality: the lazy table must not build a second copy
    of anything, and it must name the module the object is DEFINED in."""
    from lumenairy.elements import berreman as ber_mod
    from lumenairy.elements import bor as bor_mod
    from lumenairy.elements import rcwa as rcwa_mod
    from lumenairy.elements.pmm import twod as twod_mod
    assert la.RCWAStack is rcwa_mod.RCWAStack
    assert ELEM.RCWAStack is rcwa_mod.RCWAStack
    assert la.pmm_2d_order_drift is twod_mod.pmm_2d_order_drift
    assert ELEM.pmm_efficiency_2d is twod_mod.pmm_efficiency_2d
    assert la.BerremanStack is ber_mod.BerremanStack
    assert la.BORStack is bor_mod.BORStack


def test_a_resolved_lazy_name_is_cached_in_module_globals():
    """Second access must be an ordinary dict hit, not another
    ``import_module`` round trip."""
    _ = la.RCWAStack
    assert 'RCWAStack' in vars(la)
    _ = ELEM.rcwa_efficiency_1d
    assert 'rcwa_efficiency_1d' in vars(ELEM)


def test_star_import_from_elements_still_binds_every_public_name():
    ns: dict = {}
    exec('from lumenairy.elements import *', ns)          # noqa: S102
    missing = [n for n in ELEM.__all__ if n not in ns]
    assert not missing, f'star-import lost {missing}'


def test_objects_from_lazy_modules_still_pickle():
    """pickle resolves by ``__module__`` / ``__qualname__`` and imports the
    DEFINING module, so laziness at the facade must be invisible to it."""
    from lumenairy.elements.rcwa import RCWAResult
    for obj in (la.RCWAStack, la.rcwa_efficiency_1d, RCWAResult,
                la.BORStack, la.PMMStack):
        assert pickle.loads(pickle.dumps(obj)) is obj


def test_the_lazy_tables_are_complete_and_consistent():
    """Every table entry must resolve from the module it names (not from a
    facade by accident), and every top-level entry must be in ``__all__``."""
    for name, where in ELEM._LAZY_NAMES.items():
        mod = importlib.import_module(f'lumenairy.elements.{where}')
        assert getattr(mod, name) is getattr(ELEM, name), (
            f'lumenairy.elements.{name} does not come from '
            f'lumenairy.elements.{where}')
    for name, where in la._LAZY_SOLVER_NAMES.items():
        mod = importlib.import_module(f'lumenairy.{where}')
        assert getattr(mod, name) is getattr(la, name), (
            f'lumenairy.{name} does not come from lumenairy.{where}')
        assert name in la.__all__, (
            f'{name!r} is lazily resolvable but missing from '
            f'lumenairy.__all__')


def test_the_lazy_tables_cover_every_name_the_eager_blocks_bound():
    """The names ``lumenairy/__init__.py`` and ``elements/__init__.py`` used
    to bind eagerly are exactly the union of each submodule's public surface
    that the facade's ``__all__`` still advertises -- so a name dropped from
    a lazy table would show up here as an ``__all__`` entry with no source."""
    for name in ELEM.__all__:
        if name in ELEM._LAZY_NAMES:
            continue
        assert name in vars(ELEM), (
            f'lumenairy.elements.__all__ advertises {name!r}, which is '
            f'neither imported eagerly nor in _LAZY_NAMES')


# ===========================================================================
# P2-8 -- the one import-time layering violation is gone
# ===========================================================================

#: Layers that sit ABOVE ``raytrace``, per the audit's own layering table
#: (TESTS-ARCH P2-8): its rows ``raytrace -> elements`` (2 edges, 1 at module
#: level -- the violation) and ``raytrace -> propagators`` (2 edges, 0 at
#: module level) are what fix the relation.  Scoped to ``raytrace`` on
#: purpose: the elements/propagators ordering is genuinely ambiguous in this
#: library (``elements.polarization`` and ``elements._lens_real`` import
#: propagator KERNELS at module level by design), and a test that adjudicated
#: it would be asserting an opinion rather than the finding.
_ABOVE_RAYTRACE = ('elements', 'propagators', 'analysis', 'io', 'optimize')


def test_raytrace_has_no_module_level_import_of_a_higher_layer():
    """FAIL-BEFORE: on the pre-fix tree this names
    ``lumenairy.raytrace.surface:22 -> lumenairy.elements.lenses`` -- the
    single import-time layering violation of the audit's 12."""
    edges = _module_level_edges()
    bad = []
    for mod, targets in edges.items():
        if not mod.startswith('lumenairy.raytrace'):
            continue
        for target, lineno in targets:
            parts = target.split('.')
            if len(parts) > 1 and parts[1] in _ABOVE_RAYTRACE:
                bad.append(f'{mod}:{lineno} -> {target}')
    assert not bad, (
        'raytrace imports a higher layer at IMPORT time:\n  - '
        + '\n  - '.join(sorted(bad))
        + '\nMove the import into the function that needs it -- the idiom 11 '
          'of the audit\'s 12 layering edges already use, and the one '
          '``_base_surface_sag_xy`` now uses for the sag kernels.')


def test_raytrace_surface_has_no_module_level_elements_binding():
    """The specific site, pinned by name so a re-hoist is caught even if the
    layer scoping above is ever relaxed."""
    import lumenairy.raytrace.surface as surf
    assert not hasattr(surf, 'surface_sag_general')
    assert not hasattr(surf, 'surface_sag_biconic')


def test_sag_dispatch_is_bit_identical_after_the_import_moved():
    """Bit-identity gate for the mechanical move.  The oracle is
    ``elements.lenses``'s own kernels called directly -- the same functions
    the moved import binds -- so the assertion isolates the DISPATCH, which
    is the only thing the move could have changed.  ``array_equal``, not a
    tolerance: a pure re-binding has no numerical content at all.
    """
    from lumenairy.elements.lenses import (
        surface_sag_biconic,
        surface_sag_general,
    )
    from lumenairy.raytrace.surface import Surface, _surface_sag_xy

    x = np.linspace(-3e-3, 3e-3, 17)
    y = np.linspace(-2e-3, 2e-3, 17)

    rot = Surface(radius=25e-3, conic=-0.7,
                  aspheric_coeffs={4: 1.0e6, 6: -2.0e9})
    got = _surface_sag_xy(x, y, rot)
    want = surface_sag_general(x * x + y * y, rot.radius, rot.conic,
                               rot.aspheric_coeffs)
    assert np.array_equal(got, want), 'rotationally-symmetric sag dispatch'

    bic = Surface(radius=25e-3, radius_y=40e-3, conic=-0.7, conic_y=0.3)
    got_b = _surface_sag_xy(x, y, bic)
    want_b = surface_sag_biconic(
        x, y, R_x=bic.radius, R_y=bic.radius_y, conic_x=bic.conic,
        conic_y=bic.conic_y, aspheric_coeffs=bic.aspheric_coeffs,
        aspheric_coeffs_y=bic.aspheric_coeffs_y)
    assert np.array_equal(got_b, want_b), 'biconic sag dispatch'

    flat = Surface(radius=np.inf, conic=0.0)
    assert np.array_equal(
        _surface_sag_xy(x, y, flat),
        surface_sag_general(x * x + y * y, flat.radius, flat.conic, None))


def test_raytrace_import_closure_excludes_elements_lenses():
    """The layering claim made operational: the transitive module-level
    closure of ``lumenairy.raytrace.surface`` must not contain
    ``lumenairy.elements.lenses``.

    (The closure still contains ``lumenairy`` itself -- importing any
    submodule executes the package ``__init__`` -- so this is a statement
    about ``surface.py``'s own edges, which is what the finding is about.)
    """
    edges = _module_level_edges()
    seen, stack = set(), ['lumenairy.raytrace.surface']
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for target, _lineno in edges.get(mod, ()):
            if target not in seen:
                stack.append(target)
    assert 'lumenairy.elements.lenses' not in seen, (
        'raytrace.surface still reaches elements.lenses through module-level '
        'imports: ' + str(sorted(m for m in seen
                                 if m.startswith('lumenairy.elements'))))
