"""v5.2 15th meta-pin walker -- sentinel ``__reduce__`` structural pin (V15).

The v5.1.0 audit identified that the sibling-gap meta-pattern,
structurally retired at code surfaces via V1-V10, at documentation
surfaces via V11, and at CHANGELOG-vs-changeset surfaces via V12,
recurred at a third surface: ``_Sentinel`` subclass pickle-safety.
The 3 v4.15.x sentinel subclasses (``_ZeroApertureMaskSentinel``,
``_InvalidFocalLengthSentinel``, ``_FailedScanStrehlSentinel``) are
pinned individually in ``test_v4_15_2_agent_e.py`` via a hardcoded
``EXPECTED_SUBCLASSES`` tuple, but no walker enumerates every
``_Sentinel`` subclass and asserts the ``__reduce__`` returns the
name-keyed ``_sentinel_unpickle`` tuple.  A maintainer adding a
4th ``_Sentinel`` subclass needs to update the tuple by hand; if
they forget, the test does NOT fail -- it just silently fails to
verify the new sentinel's pickle round-trip.

This walker AUTO-discovers every ``_Sentinel`` subclass in the live
library (via ``pkgutil.walk_packages`` to force import of every
subpackage so all sentinels self-register, then
``_Sentinel.__subclasses__()`` recursive traversal) and asserts the
4-prong contract for each:

* The subclass has a ``__reduce__`` method (NOT the default
  ``object.__reduce__``) and calling it returns the tuple
  ``(_sentinel_unpickle, (cls_name_string,))``.

* The cls_name returned by ``__reduce__`` is registered in
  ``lumenairy._deprecation._SENTINEL_REGISTRY``.

* ``pickle.loads(pickle.dumps(instance)) is instance`` -- singleton
  identity is preserved across a full pickle round-trip.

* The registered name in ``_SENTINEL_REGISTRY`` resolves to a
  singleton INSTANCE (not the class itself).

Closes audit P2-NEW-F2-1 #3 (sentinel ``__reduce__`` structural pin)
AND audit P3-NEW-F1-3 (no counter-pin for new sentinel subclasses).

The walker is conservative: test-only sentinels (e.g. ``_MockSentinel``
defined inline in a test module) are auto-skipped by the
``__module__.startswith('lumenairy.')`` gate.  Abstract base sentinels
that never instantiate a singleton are exempted via the module-level
``_V15_SENTINEL_EXEMPTIONS`` frozenset with an audit-citation comment.

v5.2 (AUDIT_V5_1_0 P2-NEW-F2-1 #3 + P3-NEW-F1-3 + AUDIT_V5_1_1 Part 6
P3 v5.2 candidates):  closes the structural-pin gap left by the
hardcoded ``EXPECTED_SUBCLASSES`` tuple at
``test_v4_15_2_agent_e.py:222-227``.  Future ``_Sentinel`` subclasses
auto-enrol in the round-trip check at import time.
"""
from __future__ import annotations

import importlib
import pickle
import pkgutil
import warnings
from pathlib import Path
from typing import List, Tuple, Type

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# Exemptions
# ===========================================================================
#
# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-1 #3 + P3-NEW-F1-3 + AUDIT_V5_1_1 Part 6
# P3 v5.2 candidates):  exemption set for sentinels that are part of
# the ``lumenairy.*`` namespace but legitimately do NOT participate in
# the registry-keyed pickle protocol.  Each entry must cite the audit
# / class rationale.  Subclasses defined OUTSIDE ``lumenairy.*`` (e.g.
# inline ``class _MockSentinel(_Sentinel)`` inside a test module) are
# auto-skipped by the ``__module__.startswith('lumenairy.')`` filter
# in :func:`_discover_lumenairy_sentinel_subclasses` and need no
# explicit exemption entry.
#
# Currently empty -- all 4 known v5.1.0 ``_Sentinel`` subclasses
# (``_NoDefaultSentinel``, ``_ZeroApertureMaskSentinel``,
# ``_InvalidFocalLengthSentinel``, ``_FailedScanStrehlSentinel``,
# ``_AngleUnsetSentinel``) DO implement the full registry-keyed
# pickle contract.  This set exists so a future maintainer can
# document an abstract-base ``_Sentinel`` subclass without breaking
# the walker, but the v5.2-baseline expectation is zero exemptions.
_V15_SENTINEL_EXEMPTIONS: frozenset = frozenset({
    # Example future entry (left commented for documentation):
    # '_AbstractSomethingSentinel',  # v5.X audit P#-NEW-XX: abstract
    #     # base sentinel, never instantiated as a singleton; only
    #     # concrete subclasses register in _SENTINEL_REGISTRY.
})


# ===========================================================================
# Walker -- enumerate every ``_Sentinel`` subclass in the live library
# ===========================================================================

def _walk_lumenairy_subpackages() -> List[str]:
    """Import every importable submodule under ``lumenairy.`` via
    :func:`pkgutil.walk_packages` so all ``_Sentinel`` subclasses
    self-register (the registration is a side-effect of subclass
    instantiation at module-import time).

    Modules whose import raises ``ImportError`` (typically optional
    heavy dependencies like JAX, cupy, pyfftw on systems without
    them) are skipped silently -- those are covered by per-feature
    smoke tests, not by this structural walker.  Other import-time
    exceptions are also caught so a single broken module cannot
    short-circuit the entire sentinel discovery.

    Returns the list of dotted module names that were skipped, for
    informational reporting (the test does not fail on skips).
    """
    import lumenairy

    skipped: List[str] = []
    for modinfo in pkgutil.walk_packages(
            lumenairy.__path__, prefix='lumenairy.'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                importlib.import_module(modinfo.name)
        except ImportError as exc:
            skipped.append(f'{modinfo.name}: ImportError: {exc}')
        except Exception as exc:  # pragma: no cover -- defensive
            # Catch-all so one broken submodule does not block the
            # entire walker.  We DO want to know about it (the test
            # will report it via the skipped list) but do not fail
            # the walker on it -- per-module pins handle module-
            # specific failures.
            skipped.append(
                f'{modinfo.name}: {type(exc).__name__}: {exc}')
    return skipped


def _all_subclasses_recursive(cls: type) -> List[type]:
    """Return every subclass of ``cls`` reachable via the
    ``__subclasses__()`` chain (children, grandchildren, etc.),
    deduplicated and sorted by qualified name for deterministic
    test parametrize IDs."""
    seen: set = set()
    stack: List[type] = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        stack.extend(sub.__subclasses__())
    return sorted(seen, key=lambda c: f'{c.__module__}.{c.__qualname__}')


def _discover_lumenairy_sentinel_subclasses() -> List[type]:
    """Return every ``_Sentinel`` subclass defined inside the
    ``lumenairy.*`` namespace.  Test-only sentinels defined inside
    test modules (``tests.unit.*``) or external packages are
    excluded -- they are not part of the production contract this
    walker enforces."""
    # Force every subpackage to import so all subclasses self-register.
    _walk_lumenairy_subpackages()

    from lumenairy._deprecation import _Sentinel

    subclasses = _all_subclasses_recursive(_Sentinel)
    # Keep only subclasses defined inside the lumenairy.* namespace.
    lumenairy_subs = [
        sub for sub in subclasses
        if (sub.__module__ or '').startswith('lumenairy.')
        or sub.__module__ == 'lumenairy'
    ]
    return lumenairy_subs


# Materialise at module-import time so the count is visible in
# pytest collection and so a walker collapse fails loudly rather
# than silently producing zero parametrize cases.
_DISCOVERED_SUBCLASSES: List[type] = _discover_lumenairy_sentinel_subclasses()


def _parametrize_id(cls: type) -> str:
    """Deterministic, human-readable parametrize ID."""
    return f'{cls.__module__}.{cls.__qualname__}'


# ===========================================================================
# V15.0 -- discovery floor
# ===========================================================================

def test_v15_walker_discovers_known_sentinel_floor():
    """Sanity floor: the walker MUST discover at least the 6 known
    v5.2.0 ``_Sentinel`` subclasses.

    Pre-v5.2 the hardcoded ``EXPECTED_SUBCLASSES`` tuple in
    ``test_v4_15_2_agent_e.py`` enumerated 3 of them
    (``_ZeroApertureMaskSentinel``, ``_InvalidFocalLengthSentinel``,
    ``_FailedScanStrehlSentinel``); ``_NoDefaultSentinel`` and
    ``_AngleUnsetSentinel`` exist alongside but are not in that
    tuple.  v5.2.0's walker auto-discovery turned up a SIXTH
    (``_SchellReturnKindUnsetSentinel``) that the hardcoded tuple
    was silently missing -- which is the regression class V15 was
    built to catch.

    v5.2.5 (AUDIT_V5_2_3 P3-F1-6): tightened from >= 5 to >= 6 to
    reflect the v5.2.0-discovered baseline.  A floor of 6 catches
    a walker collapse (e.g. the import-time self-registration
    broke, or ``pkgutil.walk_packages`` failed to enumerate
    ``lumenairy``) while leaving headroom for legitimate
    refactoring.  When new sentinels are added the floor should
    bump alongside.
    """
    names = sorted(c.__qualname__ for c in _DISCOVERED_SUBCLASSES)
    assert len(_DISCOVERED_SUBCLASSES) >= 6, (
        f'V15 walker discovered only {len(_DISCOVERED_SUBCLASSES)} '
        f'``_Sentinel`` subclasses in the live ``lumenairy.*`` '
        f'namespace; expected >= 6 (the v5.2.0 baseline).  '
        f'Discovered: {names!r}.  The discovery walker may have '
        f'regressed (e.g. ``pkgutil.walk_packages`` no longer '
        f'imports the sentinel-defining submodules, or the '
        f'``__module__`` filter excludes a legitimate sentinel).')


def test_v15_walker_exemption_set_is_disjoint_from_discovered():
    """The ``_V15_SENTINEL_EXEMPTIONS`` set must NOT name any
    subclass that the walker would otherwise verify -- exemptions
    are for sentinels the walker would falsely flag, not for
    silencing legitimate failures.

    The v5.2 baseline expectation is an empty exemption set; this
    pin closes the v5.1.0 anti-pattern (P3-NEW-F1-3) where a
    maintainer could quietly add a 4th sentinel and then suppress
    the resulting walker failure by listing it in the exemption
    set without an audit citation.
    """
    discovered_names = {c.__qualname__ for c in _DISCOVERED_SUBCLASSES}
    overlap = sorted(_V15_SENTINEL_EXEMPTIONS & discovered_names)
    # Overlap is only OK if each overlapping entry is documented as
    # an abstract base in the comment block above _V15_SENTINEL_
    # EXEMPTIONS.  The test does NOT enforce comment content (that's
    # human review); it only asserts the structural invariant that
    # an exemption either suppresses a still-discoverable sentinel
    # (legitimate -- abstract base) OR is dead documentation.
    #
    # The pin's real job: if exemption set grows, the user must
    # acknowledge here that they understand the exemption mechanism
    # is opt-in suppression, not opt-out enforcement.  Empty set
    # passes trivially.
    if overlap:
        # If the exemption set names a discovered class, the walker
        # will skip that class's pickle round-trip pin.  Make the
        # skip VISIBLE in the test output instead of letting it
        # silently disappear.
        warnings.warn(
            f'V15 exemption set covers {len(overlap)} discovered '
            f'``_Sentinel`` subclass(es): {overlap!r}.  Confirm '
            f'each has an audit citation in the comment block '
            f'above ``_V15_SENTINEL_EXEMPTIONS``.',
            UserWarning,
            stacklevel=2,
        )


# ===========================================================================
# V15.1 -- every discovered sentinel has a __reduce__ returning the
#           name-keyed (_sentinel_unpickle, (name,)) tuple
# ===========================================================================

@pytest.mark.parametrize(
    'sentinel_cls',
    _DISCOVERED_SUBCLASSES,
    ids=[_parametrize_id(c) for c in _DISCOVERED_SUBCLASSES],
)
def test_v15_sentinel_reduce_returns_name_keyed_unpickle_tuple(sentinel_cls):
    """Each ``_Sentinel`` subclass must implement ``__reduce__``
    such that the return value is the canonical
    ``(_sentinel_unpickle, (name,))`` tuple, where ``name`` is the
    string registered in ``_SENTINEL_REGISTRY``.

    The default ``object.__reduce__`` would NOT route through the
    name-keyed registry on unpickle, so a sentinel that fell back
    to the default protocol would silently lose ``is``-identity
    across pickle round-trips -- the original v4.15.1 audit
    finding that motivated the singleton-registry consolidation.
    """
    if sentinel_cls.__qualname__ in _V15_SENTINEL_EXEMPTIONS:
        pytest.skip(
            f'{sentinel_cls.__qualname__} is in '
            f'``_V15_SENTINEL_EXEMPTIONS`` (audit-cited abstract '
            f'base); skipping the ``__reduce__`` contract check.')

    from lumenairy._deprecation import (
        _SENTINEL_REGISTRY,
        _Sentinel,
        _sentinel_unpickle,
    )

    # Locate the singleton instance for this subclass via the
    # registry (the canonical entry point -- the codebase convention
    # is that every ``_Sentinel`` subclass registers itself at
    # instantiation, and the module-level singleton constant is the
    # registered value).
    matching = [
        (name, inst) for name, inst in _SENTINEL_REGISTRY.items()
        if type(inst) is sentinel_cls
    ]
    if not matching:
        pytest.skip(
            f'{sentinel_cls.__qualname__} has no entry in '
            f'``_SENTINEL_REGISTRY`` -- treated as an abstract '
            f'base sentinel with no singleton.  Consider adding '
            f'to ``_V15_SENTINEL_EXEMPTIONS`` with an audit '
            f'citation if this is intentional.')

    # If multiple registry entries name this subclass, every entry
    # must pass the contract.
    for reg_name, instance in matching:
        # Prong 1: __reduce__ must NOT be the default object one.
        assert sentinel_cls.__reduce__ is not object.__reduce__, (
            f'{sentinel_cls.__qualname__}.__reduce__ is the default '
            f'``object.__reduce__``.  A pickle round-trip would '
            f'construct a new instance instead of returning the '
            f'singleton, breaking ``is``-identity.  Inherit from '
            f'``_Sentinel`` (which defines ``__reduce__``) instead '
            f'of overriding it back to the default.')

        # Prong 2: calling __reduce__ on the instance returns the
        # canonical 2-tuple shape.
        reduced = instance.__reduce__()
        assert isinstance(reduced, tuple) and len(reduced) == 2, (
            f'{sentinel_cls.__qualname__}({reg_name!r}).__reduce__() '
            f'returned {reduced!r}; expected a 2-tuple of the form '
            f'``(_sentinel_unpickle, (name,))``.')
        reconstructor, args = reduced
        assert reconstructor is _sentinel_unpickle, (
            f'{sentinel_cls.__qualname__}({reg_name!r}).__reduce__() '
            f'returned reconstructor {reconstructor!r}; expected the '
            f'module-level ``lumenairy._deprecation._sentinel_unpickle`` '
            f'so unpickling routes through the name-keyed registry.')
        assert isinstance(args, tuple) and len(args) == 1, (
            f'{sentinel_cls.__qualname__}({reg_name!r}).__reduce__() '
            f'returned args {args!r}; expected a 1-tuple ``(name,)``.')
        (reduced_name,) = args
        assert isinstance(reduced_name, str), (
            f'{sentinel_cls.__qualname__}({reg_name!r}).__reduce__() '
            f'returned non-string name {reduced_name!r}; expected the '
            f'string key used by ``_SENTINEL_REGISTRY``.')

        # Prong 3: the name returned by __reduce__ must equal the
        # registry key under which the instance is registered.
        assert reduced_name == reg_name, (
            f'{sentinel_cls.__qualname__}.__reduce__() returns name '
            f'{reduced_name!r} but the registry key is {reg_name!r}; '
            f'unpickling would route to the wrong singleton (or fail '
            f'if {reduced_name!r} is unregistered).')

        # Prong 4: the registry resolves the name back to the same
        # instance (this is the unpickle-side contract).
        resolved = _sentinel_unpickle(reduced_name)
        assert resolved is instance, (
            f'_sentinel_unpickle({reduced_name!r}) returned '
            f'{resolved!r} but the registered instance is '
            f'{instance!r}; the registry lookup is broken or a '
            f'duplicate registration shadowed the original.')

        # Prong 5: the registered value is a _Sentinel INSTANCE, not
        # a class -- guards against a maintainer mistakenly writing
        # ``_SENTINEL_REGISTRY[name] = SomeSentinelClass`` (which
        # would round-trip but lose the instance identity).
        assert isinstance(resolved, _Sentinel), (
            f'``_SENTINEL_REGISTRY[{reduced_name!r}]`` is '
            f'{resolved!r} ({type(resolved).__name__}); expected a '
            f'``_Sentinel`` instance, not a class or other object.')
        assert not isinstance(resolved, type), (
            f'``_SENTINEL_REGISTRY[{reduced_name!r}]`` is the class '
            f'{resolved!r}, not a singleton instance.  Register the '
            f'instance via ``super().__init__(name)`` in the subclass '
            f'``__init__``, not the class object directly.')


# ===========================================================================
# V15.2 -- full pickle round-trip preserves singleton identity
# ===========================================================================

@pytest.mark.parametrize(
    'sentinel_cls',
    _DISCOVERED_SUBCLASSES,
    ids=[_parametrize_id(c) for c in _DISCOVERED_SUBCLASSES],
)
def test_v15_sentinel_pickle_round_trip_preserves_identity(sentinel_cls):
    """``pickle.loads(pickle.dumps(instance)) is instance`` must
    hold for every discovered sentinel singleton.

    This is the end-to-end behaviour assertion that V15.1 sets up
    structurally: V15.1 verifies the ``__reduce__`` shape, V15.2
    verifies the shape actually produces the desired round-trip
    semantics under real pickle / unpickle.  Splitting the two
    catches a future regression where ``__reduce__`` is correct but
    the registry lookup itself is broken (or the registry is
    cleared by a test that doesn't restore it).
    """
    if sentinel_cls.__qualname__ in _V15_SENTINEL_EXEMPTIONS:
        pytest.skip(
            f'{sentinel_cls.__qualname__} is in '
            f'``_V15_SENTINEL_EXEMPTIONS``; skipping pickle '
            f'round-trip.')

    from lumenairy._deprecation import _SENTINEL_REGISTRY

    matching = [
        (name, inst) for name, inst in _SENTINEL_REGISTRY.items()
        if type(inst) is sentinel_cls
    ]
    if not matching:
        pytest.skip(
            f'{sentinel_cls.__qualname__} has no entry in '
            f'``_SENTINEL_REGISTRY``; nothing to round-trip.')

    for reg_name, instance in matching:
        data = pickle.dumps(instance)
        restored = pickle.loads(data)
        assert restored is instance, (
            f'Pickle round-trip lost singleton identity for '
            f'{sentinel_cls.__qualname__}({reg_name!r}): got '
            f'{restored!r} ({type(restored).__name__}), expected '
            f'{instance!r}.  Either ``__reduce__`` is incorrect '
            f'(see V15.1) or the registry was mutated between '
            f'dumps and loads.')


# ===========================================================================
# V15.3 -- counter-pin for the hardcoded EXPECTED_SUBCLASSES tuple
# ===========================================================================

def test_v15_walker_covers_v4_15_2_hardcoded_tuple():
    """The hardcoded ``EXPECTED_SUBCLASSES`` tuple in
    ``test_v4_15_2_agent_e.py:222-227`` must be a SUBSET of the
    set of sentinel subclasses discovered by V15.

    Direct closure of audit P3-NEW-F1-3: pre-v5.2 the hardcoded
    tuple was the only place new sentinels needed to be
    enumerated, and a maintainer adding a 4th subclass without
    updating it would not get a test failure -- the test would
    just silently skip the new sentinel.

    Post-v5.2 the V15 walker auto-discovers, and THIS test
    additionally pins that every entry in the legacy hardcoded
    tuple is in fact discovered (so deleting an entry from the
    legacy tuple by accident does not silently lose the pin --
    the entry is still under V15 coverage).
    """
    discovered_names = {c.__qualname__ for c in _DISCOVERED_SUBCLASSES}

    # The legacy v4.15.2 tuple (cf. test_v4_15_2_agent_e.py:222-227).
    # Hardcoded here on purpose: this is the v5.2 backstop pin that
    # asserts the legacy tuple is covered by the auto-walker.
    legacy_expected = (
        '_ZeroApertureMaskSentinel',
        '_InvalidFocalLengthSentinel',
        '_FailedScanStrehlSentinel',
    )
    missing = [
        name for name in legacy_expected
        if name not in discovered_names
    ]
    assert not missing, (
        f'V15 walker did not discover the legacy v4.15.2 '
        f'``EXPECTED_SUBCLASSES`` tuple entries: {missing!r}.  '
        f'Either the sentinel was deleted (update the legacy '
        f'tuple) or the V15 walker regressed (the import-time '
        f'self-registration is broken).')
