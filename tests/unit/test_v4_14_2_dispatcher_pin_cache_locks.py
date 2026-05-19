"""Parametrized dispatcher-pin for cache <-> lock pairing.

Audit context
-------------

``AUDIT_V4_14_1_2026_05_17.md`` P1-NEW-2 surfaced a meta-finding:
v4.14.1 added ``threading.Lock`` instances to 3 caches introduced
during that same v4.14.0/v4.14.1 cycle
(``_LG_MODE_STACK_LOCK``, ``_HG_MODE_STACK_LOCK``,
``_WRAPPER_MERIT_CACHE_LOCK``) plus the pre-existing
``_ASM_CACHE_LOCK``, but missed 7 older module-level
``OrderedDict`` caches that were equally susceptible to torn
read-modify-write sequences under concurrent threads.

v4.14.2 / Agent C extends the v4.14.1 ``test_v4_14_1_dispatcher_
pin_cache_clears.py`` pattern to a second meta-invariant:

* every module-level ``_<CACHENAME>_CACHE`` (caught by regex on
  the variable name itself) MUST have a corresponding
  ``_<CACHENAME>_LOCK`` in the same module namespace.

* the reverse direction: every ``_<CACHENAME>_LOCK`` (excluding a
  small list of legitimate non-cache locks documented below) MUST
  have a corresponding ``_<CACHENAME>_CACHE`` partner.

The reverse check protects against a future ``_FOO_LOCK`` being
added without its companion cache (i.e. a lock created for state
that should have been a cache but was scattered across module
globals instead) AND against a cache being renamed without updating
its companion lock.

Legitimate non-cache locks (the reverse-direction exemption list):

* ``_PYFFTW_PLAN_LOCK`` -- guards the pyFFTW plan cache + bad
  shapes; the pyFFTW *plan cache* is named ``_PYFFTW_PLAN_CACHE``
  so this pair is matched.
* ``_ZARR_MKDIR_LOCK`` (per AUDIT spec) -- guards an *operation*
  (mkdir) not a cache.  Allowed by name.

A future feature that adds a new ``^_.*_CACHE$`` global will fail
this test at the parametrized entry point until the companion
``_<...>_LOCK`` lands in the same module -- closing the P1-NEW-2-
class regression at CI rather than at the next audit.

Author: Andrew Traverso -- v4.14.2 / Agent C
"""
from __future__ import annotations

import importlib
import pkgutil
import re
import threading
from typing import List, Tuple

import pytest

import lumenairy as la


# ============================================================================
# Submodule walker
# ============================================================================

_CACHE_PATTERN = re.compile(r'^_.*_CACHE$')
_LOCK_PATTERN = re.compile(r'^_.*_LOCK$')

# Reverse-direction exemption list: locks that legitimately do NOT
# correspond to a cache (they guard an operation or a non-cache piece
# of state).  Lock names that match the ``_.*_PATCH_LOCK`` pattern
# guard a monkey-patch operation (per audit spec) and are exempt by
# pattern -- documented at the regex below.
_LOCK_WITHOUT_CACHE_EXEMPTIONS = frozenset({
    # ``_ZARR_MKDIR_LOCK`` (audit spec) -- if it lands as a bare
    # ``_LOCK`` (no ``_PATCH`` infix), guards a mkdir operation
    # not a cache.
    '_ZARR_MKDIR_LOCK',
    # ``_PYFFTW_PLAN_LOCK`` matches ``_PYFFTW_PLAN_CACHE``, so it's
    # NOT an exemption -- listed here as documentation of the
    # canonical pair to clarify the walker's intent.  Kept out of
    # the actual frozenset.
    # v4.16.0 (ROADMAP #15): ``_REGISTRY_LOCK`` in
    # ``lumenairy._cache_registry`` guards the central cache-clearer
    # registry (a ``_CACHE_CLEARERS`` dict of name -> callable, NOT
    # a numerical / array cache).  The dict's name intentionally
    # doesn't end in ``_CACHE`` because the dispatcher pin in this
    # test would then treat it as a real numerical cache and demand
    # an entry-clear discipline that doesn't apply to a registry of
    # clear-functions.  The lock is real (threading.Lock) and the
    # state it guards is the registry membership itself; documenting
    # here that the pair is intentional.
    '_REGISTRY_LOCK',
})

# Pattern-based exemption: ``_<X>_PATCH_LOCK`` guards an in-place
# monkey-patch (per AUDIT_V4_14_1 spec).  These locks have no
# companion cache by design.  Currently matches:
# * ``_ZARR_MKDIR_PATCH_LOCK`` (lumenairy/io/storage.py)
_PATCH_LOCK_PATTERN = re.compile(r'^_.*_PATCH_LOCK$')


def _strip_lock_suffix(name: str) -> str:
    """Map ``_FOO_LOCK`` -> ``_FOO``.  Returns the input on miss."""
    return name[:-len('_LOCK')] if name.endswith('_LOCK') else name


def _strip_cache_suffix(name: str) -> str:
    """Map ``_FOO_CACHE`` -> ``_FOO``.  Returns the input on miss."""
    return name[:-len('_CACHE')] if name.endswith('_CACHE') else name


def _discover_cache_lock_pairs() -> Tuple[
        List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Walk every submodule under ``lumenairy`` and collect:

    1. Every ``_<...>_CACHE`` global -> list of (module, cache_name).
    2. Every ``_<...>_LOCK``  global -> list of (module, lock_name).

    Returns both lists, sorted for deterministic CI ordering.
    """
    caches: List[Tuple[str, str]] = []
    locks: List[Tuple[str, str]] = []
    for modinfo in pkgutil.walk_packages(
            la.__path__, prefix='lumenairy.'):
        try:
            mod = importlib.import_module(modinfo.name)
        except ImportError:
            # Optional-dep modules (e.g. JAX-only files) may fail to
            # import in environments without that dependency.  Skip
            # rather than failing the meta-pin -- per-cache smoke
            # tests cover the JAX-gated paths.
            continue
        for name in dir(mod):
            if _CACHE_PATTERN.match(name):
                # The C.4 contract is about real cache containers
                # (dict / OrderedDict / set), not constants like
                # ``_FOO_CACHE_SIZE`` (the regex ``_.*_CACHE$``
                # already excludes those -- they end in _SIZE).
                caches.append((modinfo.name, name))
            elif _LOCK_PATTERN.match(name):
                # Must actually be a threading.Lock-ish object.
                obj = getattr(mod, name)
                if isinstance(obj, type(threading.Lock())):
                    locks.append((modinfo.name, name))
    return sorted(caches), sorted(locks)


_DISCOVERED_CACHES, _DISCOVERED_LOCKS = _discover_cache_lock_pairs()


# ============================================================================
# Pre-flight -- the walker MUST find at least one cache and one lock
# ============================================================================

def test_walker_found_caches():
    """If the walker returns no caches the parametrize lists below
    collapse to zero tests and the meta-pin silently no-ops --
    exactly the failure mode P1-NEW-2 was meant to catch."""
    assert len(_DISCOVERED_CACHES) > 0, (
        'Cache walker found ZERO ``_.*_CACHE`` globals.  Either the '
        'walker is broken or the library no longer has any module-'
        'level caches -- both are bugs that would let P1-NEW-2-class '
        'regressions slip past unnoticed.')


def test_walker_found_locks():
    """Same idea for locks."""
    assert len(_DISCOVERED_LOCKS) > 0, (
        'Lock walker found ZERO ``_.*_LOCK`` globals.  This means '
        'the library has lost its thread-safety locks entirely OR '
        'the walker is broken.')


# ============================================================================
# Pin 1 -- every cache has a companion lock in the same module
# ============================================================================

# Caches that intentionally share a single lock with sibling caches
# in the same module.  Each entry maps a cache name to the
# alternative-named lock that actually guards it.  This is the
# "one lock guards many caches" pattern (e.g. ``_ASM_CACHE_LOCK``
# guards ``_FREQ_GRID_CACHE``, ``_BANDLIMIT_CACHE``, ``_H_CACHE``,
# ``_PYFFTW_BAD_SHAPES``).  Entry format: full dotted module name
# plus cache name -> guarding lock name.
_SHARED_LOCK_MAPPING = {
    ('lumenairy.propagators.propagation', '_FREQ_GRID_CACHE'):
        '_ASM_CACHE_LOCK',
    ('lumenairy.propagators.propagation', '_BANDLIMIT_CACHE'):
        '_ASM_CACHE_LOCK',
    ('lumenairy.propagators.propagation', '_H_CACHE'):
        '_ASM_CACHE_LOCK',
}


def _candidate_lock_names_for_cache(cache_name: str) -> List[str]:
    """Generate the two name conventions used in this codebase.

    * ``_FOO_CACHE`` paired with ``_FOO_LOCK``  (strips _CACHE)
    * ``_FOO_CACHE`` paired with ``_FOO_CACHE_LOCK``  (keeps _CACHE)

    Both conventions exist in the library; the meta-pin accepts
    either.  We return both candidates so the test can hasattr-
    walk them.
    """
    base = _strip_cache_suffix(cache_name)
    return [f'{base}_LOCK', f'{cache_name}_LOCK']


# Known sibling-gaps discovered by this meta-pin but OUT OF SCOPE
# for the v4.14.2 / Agent C fix.  Each entry is a
# (module_dotted_name, cache_name) tuple.  These are xfailed (not
# skipped) so the meta-pin records them as expected failures; a
# future agent that closes one will get an XPASS notification.
#
# v4.14.2 integration closed the previously-known sibling-gap:
# ``lumenairy.propagators.asymptotic._JAX_IFT_SOLVER_CACHE`` was a
# single-cell lazy-init cache without a lock; v4.14.2 added
# ``_JAX_IFT_SOLVER_CACHE_LOCK`` and refactored the build into a
# worker function under double-check locking.  Sibling-gap set is
# now empty; future regressions land here.
_KNOWN_CACHE_SIBLING_GAPS = frozenset()


@pytest.mark.parametrize('submod_name, cache_name', _DISCOVERED_CACHES)
def test_cache_has_companion_lock(submod_name, cache_name, request):
    """Every module-level ``_<CACHENAME>_CACHE`` must have a
    corresponding lock in the same module namespace.

    Accepts EITHER naming convention -- the library uses both:

    * ``_FOO_CACHE`` paired with ``_FOO_LOCK`` (strips _CACHE)
    * ``_FOO_CACHE`` paired with ``_FOO_CACHE_LOCK`` (keeps _CACHE)

    Also accepts shared-lock pairings via
    ``_SHARED_LOCK_MAPPING`` (one lock guards many caches, e.g.
    ``_ASM_CACHE_LOCK`` for the freq/bandlimit/H caches).

    Pre-fix the v4.14.1 audit P1-NEW-2 caught 7 unlocked caches
    that pre-dated the v4.14.0/v4.14.1 lock sweep.  A future
    ``_FOO_CACHE`` that lands without its companion lock will trip
    this test at the parametrized entry point.
    """
    if (submod_name, cache_name) in _KNOWN_CACHE_SIBLING_GAPS:
        request.applymarker(pytest.mark.xfail(
            reason=f'{submod_name}.{cache_name} is a documented '
                   f'sibling-gap out of v4.14.2/Agent-C scope.  '
                   f'See ``_KNOWN_CACHE_SIBLING_GAPS`` for details.',
            strict=True))
    mod = importlib.import_module(submod_name)
    # First check the shared-lock mapping for known multi-cache
    # locks (e.g. ``_ASM_CACHE_LOCK``).
    shared = _SHARED_LOCK_MAPPING.get((submod_name, cache_name))
    if shared is not None:
        assert hasattr(mod, shared), (
            f'{submod_name}.{cache_name} is mapped to shared lock '
            f'{shared!r} but that lock is missing from the module.')
        lock_obj = getattr(mod, shared)
        assert isinstance(lock_obj, type(threading.Lock())), (
            f'{submod_name}.{shared} is not a threading.Lock; got '
            f'{type(lock_obj).__name__}.')
        return

    # Otherwise expect EITHER name convention.
    candidates = _candidate_lock_names_for_cache(cache_name)
    matches = [name for name in candidates if hasattr(mod, name)]
    assert matches, (
        f'{submod_name}.{cache_name} has NO companion lock in the '
        f'same module namespace.  Expected one of {candidates!r}.  '
        f'Caches need locks to be safe under concurrent reader-'
        f'writer access.  Add the lock at module level and wrap '
        f'every read-modify-write site (get / __setitem__ / '
        f'move_to_end / popitem(last=False)).  See '
        f'``propagators/asymptotic.py:_LG_MODE_STACK_LOCK`` for '
        f'the canonical pattern, or add the pair to '
        f'``_SHARED_LOCK_MAPPING`` if a single lock guards '
        f'multiple caches.')
    # All matches must be real threading.Lock instances.
    for name in matches:
        lock_obj = getattr(mod, name)
        assert isinstance(lock_obj, type(threading.Lock())), (
            f'{submod_name}.{name} is not a threading.Lock '
            f'instance; got {type(lock_obj).__name__}.  Cache '
            f'locks must be plain threading.Lock objects so the '
            f'with-block discipline is uniform across the '
            f'library.')


# ============================================================================
# Pin 2 -- every lock has a companion cache (or is on the exemption list)
# ============================================================================

# Locks that legitimately guard multiple caches at once.  The
# reverse-direction pin lets these match if AT LEAST ONE cache
# in the same module is mapped to this lock via
# ``_SHARED_LOCK_MAPPING``.  Distinct from the
# ``_LOCK_WITHOUT_CACHE_EXEMPTIONS`` set, which is for locks that
# guard NO cache (operations / non-cache state).
_SHARED_LOCK_NAMES = frozenset({
    name for ((_mod, _cache), name) in _SHARED_LOCK_MAPPING.items()
})


def _candidate_cache_names_for_lock(lock_name: str) -> List[str]:
    """Generate cache-name candidates for a given lock under both
    library naming conventions.

    * ``_FOO_LOCK`` (where ``_FOO_CACHE`` would be the natural
      cache name) -> generate ``_FOO_CACHE``.
    * ``_FOO_CACHE_LOCK`` (where the lock keeps the _CACHE suffix
      explicit) -> generate ``_FOO_CACHE``.

    We return both candidates so the test can hasattr-walk them.
    """
    base = _strip_lock_suffix(lock_name)
    # Convention A: ``_FOO_LOCK`` -> ``_FOO_CACHE``
    cand_a = f'{base}_CACHE'
    # Convention B: ``_FOO_CACHE_LOCK`` -> ``_FOO_CACHE``
    cand_b = base if base.endswith('_CACHE') else None
    out = [cand_a]
    if cand_b is not None and cand_b != cand_a:
        out.append(cand_b)
    return out


@pytest.mark.parametrize('submod_name, lock_name', _DISCOVERED_LOCKS)
def test_lock_has_companion_cache_or_is_exempt(submod_name, lock_name):
    """Every module-level ``_<LOCKNAME>_LOCK`` must have a
    companion cache in the same module OR be on the
    ``_LOCK_WITHOUT_CACHE_EXEMPTIONS`` allow-list OR be a known
    shared-lock that guards multiple caches.

    Catches the opposite failure mode from pin 1: a lock created
    for state that should have been a cache (i.e. scattered across
    module globals instead of consolidated into a dict) OR a cache
    that was renamed without updating its companion lock.
    """
    if lock_name in _LOCK_WITHOUT_CACHE_EXEMPTIONS:
        pytest.skip(
            f'{lock_name} is on the exemption list -- it guards an '
            f'operation or non-cache state and is allowed to exist '
            f'without a paired cache.')
    if _PATCH_LOCK_PATTERN.match(lock_name):
        # ``_<X>_PATCH_LOCK`` guards a monkey-patch operation per
        # the audit spec; no companion cache is expected.
        pytest.skip(
            f'{lock_name} matches ``_<X>_PATCH_LOCK`` -- guards a '
            f'monkey-patch operation, not a cache.  Documented '
            f'exemption per AUDIT_V4_14_1 spec.')
    # Shared-lock case: at least one cache in the same module
    # must map to this lock via _SHARED_LOCK_MAPPING.
    if lock_name in _SHARED_LOCK_NAMES:
        guarded = [c for ((m, c), L) in _SHARED_LOCK_MAPPING.items()
                   if m == submod_name and L == lock_name]
        assert guarded, (
            f'{submod_name}.{lock_name} is listed as a shared-lock '
            f'but has no caches mapped to it in this module.')
        return
    mod = importlib.import_module(submod_name)
    candidates = _candidate_cache_names_for_lock(lock_name)
    matches = [name for name in candidates if hasattr(mod, name)]
    assert matches, (
        f'{submod_name}.{lock_name} has NO companion cache in the '
        f'same module namespace.  Expected one of {candidates!r}.  '
        f'Either rename the lock to match its cache, add the cache '
        f'(if the lock guards state that should be consolidated), '
        f'add {lock_name!r} to ``_LOCK_WITHOUT_CACHE_EXEMPTIONS`` '
        f'in this test file with a comment explaining what the '
        f'lock guards, or add a shared-lock entry to '
        f'``_SHARED_LOCK_MAPPING``.')


# ============================================================================
# Pin 3 -- direct regression pin for the v4.14.2 P1-NEW-2 set
# ============================================================================

# After the v4.14.2 / Agent C fix lands, the 7 caches below MUST
# each have a paired lock.  These are name-anchored so a grep for
# the cache name lands here when investigating any future
# regression that drops one of these locks.
_V4_14_2_EXPECTED_PAIRS = [
    ('lumenairy.analysis.core',
     '_ZERNIKE_BASIS_CACHE', '_ZERNIKE_BASIS_CACHE_LOCK'),
    ('lumenairy.analysis.through_focus',
     '_THROUGH_FOCUS_SCAN_JAX_CACHE',
     '_THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK'),
    ('lumenairy.analysis.phase_retrieval',
     '_GS_KERNEL_CACHE', '_GS_KERNEL_CACHE_LOCK'),
    ('lumenairy.analysis.phase_retrieval',
     '_ER_KERNEL_CACHE', '_ER_KERNEL_CACHE_LOCK'),
    ('lumenairy.analysis.phase_retrieval',
     '_HIO_KERNEL_CACHE', '_HIO_KERNEL_CACHE_LOCK'),
    ('lumenairy.system',
     '_PROPAGATE_SYSTEM_JAX_CACHE',
     '_PROPAGATE_SYSTEM_JAX_CACHE_LOCK'),
    ('lumenairy.raytrace.jax_trace',
     '_TRACE_JAX_CACHE', '_TRACE_JAX_CACHE_LOCK'),
]


@pytest.mark.parametrize(
    'submod_name, cache_name, lock_name', _V4_14_2_EXPECTED_PAIRS)
def test_v4_14_2_expected_cache_lock_pairs(
        submod_name, cache_name, lock_name):
    """Direct name-anchored regression pin for the v4.14.2 P1-NEW-2
    cache-lock pairing fix.

    The 7 pairs below correspond exactly to the audit's listed
    sites.  Redundant with the parametrized walker above, but kept
    so a grep for any of these cache names lands here when
    investigating any future regression.
    """
    mod = importlib.import_module(submod_name)
    assert hasattr(mod, cache_name), (
        f'expected cache {submod_name}.{cache_name} not found')
    assert hasattr(mod, lock_name), (
        f'expected lock {submod_name}.{lock_name} not found '
        f'(paired with cache {cache_name!r})')
    lock_obj = getattr(mod, lock_name)
    assert isinstance(lock_obj, type(threading.Lock())), (
        f'{submod_name}.{lock_name} is not a threading.Lock '
        f'instance; got {type(lock_obj).__name__}')


# ============================================================================
# Pin 4 -- combined count check (all 10 cache+lock pairs accounted for)
# ============================================================================

def test_total_cache_count_at_least_ten():
    """Sanity check on the discovered count: there should be at
    least 10 cache-lock pairs across the library:

    - 3 v4.14.1 pairs: ``_LG_MODE_STACK``, ``_HG_MODE_STACK``,
      ``_WRAPPER_MERIT_CACHE``.
    - 1 pre-v4.14.0 pair: ``_ASM_CACHE_LOCK`` (guards multiple
      caches: ``_FREQ_GRID_CACHE``, ``_BANDLIMIT_CACHE``, ``_H_CACHE``,
      ``_PYFFTW_BAD_SHAPES``).
    - 1 pyFFTW pair: ``_PYFFTW_PLAN_CACHE`` / ``_PYFFTW_PLAN_LOCK``.
    - 7 v4.14.2 pairs (the Agent C fix list).

    Not every cache has its OWN lock (``_ASM_CACHE_LOCK`` is
    shared across 4 caches) so a strict 1:1 check would over-
    specify.  The lower bound below is conservative.
    """
    assert len(_DISCOVERED_CACHES) >= 10, (
        f'Expected at least 10 ``_.*_CACHE`` globals, got '
        f'{len(_DISCOVERED_CACHES)}.  Either caches were lost or '
        f'the walker is broken.')
    assert len(_DISCOVERED_LOCKS) >= 5, (
        f'Expected at least 5 ``_.*_LOCK`` globals (some locks '
        f'guard multiple caches), got {len(_DISCOVERED_LOCKS)}.')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
