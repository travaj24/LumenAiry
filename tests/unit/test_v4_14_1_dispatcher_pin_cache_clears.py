"""Parametrized dispatcher-pin for cache-clear helper consistency.

Audit context
-------------

``AUDIT_V4_14_0_2026_05_17.md`` P1-NEW-4 surfaced a meta-finding: the
v4.14.0 CHANGELOG advertised ``clear_lg_mode_stack_cache()`` as a
public top-level helper, and it was listed in
``lumenairy/propagators/asymptotic.__all__``, but
``lumenairy/__init__.py`` neither imported nor re-exported it.  Seven
sibling ``clear_*`` helpers (``clear_asm_caches``,
``clear_zernike_basis_cache``, ``clear_lg_polynomial_cache``,
``clear_through_focus_scan_jax_cache``, ``clear_trace_jax_cache``,
``clear_propagate_system_jax_cache``, ``clear_phase_retrieval_caches``)
were all correctly re-exported; ``clear_lg_mode_stack_cache`` slipped
through the cracks because each one had been added by a different
v4.x.y sweep and no parametrized pin existed to catch the sibling
gap.

This file enumerates every ``clear_*`` name that lives in a
submodule's ``__all__`` and asserts each is callable from the top-
level ``lumenairy`` namespace.  A future feature that adds a
``clear_X_cache`` helper to a submodule's ``__all__`` will fail this
test until the corresponding top-level import + ``__all__`` entry
lands -- closing the P1-NEW-4-class regression at CI rather than at
the next audit.

We deliberately scope the pin to "names appearing in a submodule's
``__all__``" rather than "every ``def clear_*`` found by source-walk"
because submodules sometimes define internal cache-clear helpers
(e.g. private dispatchers) that are NOT intended for the public
surface.  The ``__all__`` declaration is the canonical signal that a
helper IS public; that's the right gate to mirror at top level.

Author: Andrew Traverso -- v4.14.1 / Agent D
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from typing import List, Tuple

import pytest

import lumenairy as la

# ============================================================================
# Submodule walker
# ============================================================================

_CLEAR_PATTERN = re.compile(r'^clear_.*$')


def _discover_clear_names_in_submodule_alls() -> List[Tuple[str, str]]:
    """Walk every submodule under ``lumenairy`` and collect every
    name in its ``__all__`` that matches ``^clear_.*$``.

    Returns a list of ``(submodule_dotted_name, clear_function_name)``
    pairs for parametrization.  Sorted for deterministic CI ordering.
    """
    pairs: List[Tuple[str, str]] = []
    for modinfo in pkgutil.walk_packages(
            la.__path__, prefix='lumenairy.'):
        # Skip private modules (leading underscore on any path
        # component) -- they are not part of the public surface.
        parts = modinfo.name.split('.')
        if any(p.startswith('_') for p in parts[1:]):
            continue
        try:
            mod = importlib.import_module(modinfo.name)
        except ImportError:
            # Optional-dep modules (e.g. JAX-only files) may fail to
            # import in environments without that dependency.  Skip
            # rather than failing the meta-pin -- the JAX-gated
            # helpers are covered by per-helper smoke tests
            # elsewhere.
            continue
        names = getattr(mod, '__all__', [])
        for name in names:
            if _CLEAR_PATTERN.match(name):
                pairs.append((modinfo.name, name))
    return sorted(pairs)


# Materialise the parametrize list at module-import time so the
# test IDs are visible in collection output (and so collection
# itself fails loudly if the walker breaks).
_CLEAR_NAME_PAIRS = _discover_clear_names_in_submodule_alls()


# Sanity: the walker MUST find at least one pair.  If it returns
# empty the parametrize list collapses to zero tests and the meta-
# pin would silently no-op -- which is exactly the failure mode the
# pin is meant to prevent.  Treat that as a collection-time error.
def test_walker_found_at_least_one_clear_helper():
    """Pre-flight: the submodule walker found at least one
    ``clear_*`` entry.  Empty would mean the meta-pin below silently
    passes with zero parametrized cases -- a worse outcome than the
    P1-NEW-4 gap it's meant to catch.
    """
    assert len(_CLEAR_NAME_PAIRS) > 0, (
        "Submodule walker found ZERO clear_* entries in any "
        "submodule's __all__.  Either the walker is broken or the "
        "library no longer declares any cache-clear helpers as "
        "public -- both are bugs that would let P1-NEW-4-class "
        "regressions slip past unnoticed.")


# ============================================================================
# Pin 1 -- submodule clear_* names are all re-exported at top level
# ============================================================================

@pytest.mark.parametrize('submod_name, clear_name', _CLEAR_NAME_PAIRS)
def test_submodule_clear_helper_reexported_at_top_level(
        submod_name, clear_name):
    """Each ``clear_*`` name declared in a submodule's ``__all__``
    MUST be re-exported by the top-level ``lumenairy`` package.

    Pre-fix the v4.14.0 audit P1-NEW-4 caught ``clear_lg_mode_stack_
    cache`` slipping through this gate.  A future ``clear_X_cache``
    that lands in a submodule's ``__all__`` but is forgotten in
    ``lumenairy/__init__.py`` will trip this test at the
    parametrized entry point.
    """
    # The name must be present at the top level.
    assert hasattr(la, clear_name), (
        f"{submod_name}.__all__ exports {clear_name!r} but it is "
        f"not present as a top-level attribute of ``lumenairy``.  "
        f"Add ``from .{'.'.join(submod_name.split('.')[1:])} import "
        f"{clear_name}`` to lumenairy/__init__.py and append "
        f"{clear_name!r} to the top-level __all__.  (This is the "
        f"v4.14.0 audit P1-NEW-4 sibling-gap pattern.)")
    # And in the top-level __all__.
    assert clear_name in la.__all__, (
        f"{clear_name!r} is importable as ``la.{clear_name}`` but "
        f"not listed in ``lumenairy.__all__``; this means "
        f"``from lumenairy import *`` would miss it.  Append it to "
        f"the top-level __all__ block.")
    # And callable.
    helper = getattr(la, clear_name)
    assert callable(helper), (
        f"la.{clear_name} is present but not callable; got "
        f"{type(helper).__name__}.  Re-exports of clear-cache "
        f"helpers must be the function itself, not a documentation "
        f"string or alias.")


# ============================================================================
# Pin 2 -- the top-level clear_* names all match their submodule home
# ============================================================================

def _enumerate_top_level_clear_names() -> List[str]:
    """Every ``clear_*`` name in ``lumenairy.__all__``."""
    return sorted(
        name for name in la.__all__ if _CLEAR_PATTERN.match(name))


_TOP_LEVEL_CLEAR_NAMES = _enumerate_top_level_clear_names()


@pytest.mark.parametrize('clear_name', _TOP_LEVEL_CLEAR_NAMES)
def test_top_level_clear_helper_is_callable(clear_name):
    """The reverse check: every ``clear_*`` name listed in
    ``lumenairy.__all__`` must be importable and callable.

    Pre-fix the v4.14.0 audit found the ``__all__`` listing was
    out-of-sync with the actual imports (the listing claimed
    ``clear_lg_mode_stack_cache`` but the import was missing).  This
    catches the opposite failure mode -- a name in ``__all__`` that
    doesn't resolve to a real callable.
    """
    assert hasattr(la, clear_name), (
        f"{clear_name!r} appears in ``lumenairy.__all__`` but "
        f"``hasattr(la, {clear_name!r})`` is False -- the import "
        f"line is missing or the name was typo'd.")
    helper = getattr(la, clear_name)
    assert callable(helper), (
        f"la.{clear_name} resolves but is not callable (got "
        f"{type(helper).__name__}).")


# ============================================================================
# Pin 3 -- direct regression pin for P1-NEW-4 itself
# ============================================================================

def test_clear_lg_mode_stack_cache_pinned_directly():
    """Direct regression pin for the v4.14.0 audit P1-NEW-4 gap:
    ``clear_lg_mode_stack_cache`` is callable as ``la.clear_lg_mode_
    stack_cache``.

    Redundant with the parametrized walker above, but kept as a
    name-anchored regression pin so a grep for
    ``clear_lg_mode_stack_cache`` lands here when investigating any
    future regression.
    """
    assert hasattr(la, 'clear_lg_mode_stack_cache')
    assert callable(la.clear_lg_mode_stack_cache)
    assert 'clear_lg_mode_stack_cache' in la.__all__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
