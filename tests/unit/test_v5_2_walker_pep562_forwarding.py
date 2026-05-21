"""v5.2 14th meta-pin walker -- PEP-562 forwarding completeness (V14).

The v5.1.0 audit identified that the sibling-gap meta-pattern,
structurally retired at code surfaces via V1-V10, at documentation
surfaces via V11, at CHANGELOG-vs-changeset via V12, and at sentinel
``__reduce__`` via V13, recurred at a new surface in the v5.1.0
file-split shell: **PEP-562 ``__getattr__`` forwarding completeness**.

The v5.1.0 split moved the FFT backend infrastructure out of
``lumenairy/propagators/propagation.py`` into the sibling submodule
``lumenairy/propagators/fft_infra.py``.  Several module-level globals
in ``fft_infra`` are MUTABLE: setter functions (``set_fft_threads``,
``set_pyfftw_planner``, ``reset_fft_backend``, ...) REBIND the name to
a fresh value rather than mutating an existing container in-place.

To preserve back-compat for consumers reading
``propagation.X`` (the legacy attribute path), the shell module
defines a PEP-562 ``__getattr__`` that forwards lookups to
``fft_infra`` for a hand-maintained whitelist
``_LIVE_FORWARD_NAMES``.  The shell ALSO deletes the
``from .fft_infra import X`` import-time snapshots for every name in
the whitelist so attribute lookup falls through to ``__getattr__``
instead of returning a stale snapshot.

The audit finding (``AUDIT_V5_1_0_2026_05_20.md`` P2-NEW-F2-1 #2):

    "PEP-562 ``__getattr__`` forwarding completeness -- hand-
    maintained whitelist.  ``propagation.py:230`` has
    ``_LIVE_FORWARD_NAMES`` with 22 names.  No walker compares
    submodule mutable globals to the whitelist.  F1's P3 finding
    (``_PYFFTW_BAD_SHAPES``) is precisely an instance -- caught
    only by F1's manual review, not by any test."

The v5.1.1 closure added ``_PYFFTW_BAD_SHAPES`` to the whitelist by
hand; V14's job is to prevent the next omission of the same class.

What V14 walks
--------------

Two-directional drift check between
``lumenairy.propagators.fft_infra``'s mutable module-level globals and
``lumenairy.propagators.propagation._LIVE_FORWARD_NAMES``:

* **Forward direction** -- every mutable global in ``fft_infra`` (a
  name that is REBOUND somewhere other than its initial module-level
  assignment, OR appears in a ``global X`` declaration inside a
  function body) must appear in ``_LIVE_FORWARD_NAMES``.  An omission
  is a stale-snapshot bug: consumer code reading
  ``propagation.X`` after a setter call observes the import-time
  value rather than the current one.  This is the exact class of bug
  the v5.1.1 closure fixed for ``_PYFFTW_BAD_SHAPES``.

* **Reverse direction** -- every name in ``_LIVE_FORWARD_NAMES`` must
  resolve via ``getattr(propagation, name)`` without raising.  This
  catches the opposite drift: a name removed from ``fft_infra`` but
  left in the whitelist would still pass the forward check (no
  mutable global to compare to) but raise ``AttributeError`` at
  consumer call time.

Mutable-vs-rebound distinction
------------------------------

Container mutation (``X.clear()``, ``X[k] = v``, ``X.add(...)``,
``X.popitem()``) is NOT a rebind -- the name still points at the
same object, so the import-time snapshot in the shell stays correct.

Only REBINDS (``X = <expr>``, ``global X; X = <expr>``,
``globals()['X'] = <expr>``) require live-forwarding.  V14
distinguishes the two by inspecting AST node types: only
``ast.Assign`` (with ``ast.Name`` target), ``ast.AugAssign``, and
``ast.Global`` declarations count as evidence of mutability;
``ast.Attribute`` and ``ast.Subscript`` targets are ignored
(in-place mutation, not rebind).

Closes audit P2-NEW-F2-1 #2 (PEP-562 forwarding completeness).
"""
# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-1 #2 + AUDIT_V5_1_1 Part 6 P3 v5.2 candidates):
# this file is the V14 walker.  Symmetric to V12 (CHANGELOG-vs-
# changeset) and V13 (sentinel __reduce__): each closes one sibling-
# gap surface introduced by the v5.1.0 file-split refactor.
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo-root derivation -- match the V11 / V12 walker idiom so all three
# walkers anchor consistently regardless of pytest invocation cwd.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FFT_INFRA_PY = _REPO_ROOT / 'lumenairy' / 'propagators' / 'fft_infra.py'
_PROPAGATION_PY = _REPO_ROOT / 'lumenairy' / 'propagators' / 'propagation.py'


# ---------------------------------------------------------------------------
# V14 exemptions: names that the walker would flag as "rebound" but
# which are NOT live-state for the snapshot-staleness purpose, with
# the legitimate reason inline.  Add to this set only with explicit
# audit cover -- the default-deny posture is what makes V14 useful.
# ---------------------------------------------------------------------------

_V14_FORWARDING_EXEMPTIONS = frozenset({
    # Module-level latches that gated the v4.16.2 / v5.0.x
    # "API-only; consumer wiring follows" UserWarnings.  v5.1.0
    # retired both warnings and these latches are now permanently
    # pinned to True (see fft_infra.py:303-304).  They are formally
    # rebound by an assignment but no setter in the live API touches
    # them post-v5.1.0, so a stale snapshot on the shell cannot
    # diverge from fft_infra's current value.  Documented here so the
    # exemption is a deliberate, auditable choice rather than silent
    # walker noise.
    '_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED',
    '_DEFAULT_DY_NO_CONSUMER_WARNED',
})


# ---------------------------------------------------------------------------
# Helpers -- AST walk to discover mutable module-level globals
# ---------------------------------------------------------------------------

def _parse_fft_infra():
    """Return the parsed ``ast.Module`` for fft_infra.py.

    A pytest-fail (not pytest-skip) is appropriate if the file is
    missing: V14's whole purpose is to walk that file, and a missing
    file is a P0 structural regression in its own right.
    """
    assert _FFT_INFRA_PY.is_file(), (
        f'V14 walker cannot locate fft_infra.py at {_FFT_INFRA_PY}.  '
        f'The v5.1.0 file-split contract requires this submodule '
        f'exist; if it has been deleted, the entire propagation shell '
        f'is broken.')
    source = _FFT_INFRA_PY.read_text(encoding='utf-8')
    return ast.parse(source, filename=str(_FFT_INFRA_PY))


def _module_level_assigned_names(tree):
    """Return dict {name: [lineno, ...]} of names assigned at the
    TOP level of the module (not inside a function or class).

    Only counts ``ast.Assign`` with an ``ast.Name`` target.  An
    ``ast.AnnAssign`` (e.g. ``X: dict = ...``) also counts as
    initial assignment.  Tuple-unpacking targets (``a, b = ...``)
    are flattened.
    """
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for nm in _name_targets(tgt):
                    out.setdefault(nm, []).append(node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.setdefault(node.target.id, []).append(node.lineno)
    return out


def _name_targets(target):
    """Yield bare-name target strings from an ast.Assign target.

    Skips Attribute / Subscript targets (those are in-place mutations,
    not name rebinds).  Recurses into Tuple / List for unpacking.
    """
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _name_targets(elt)
    # Attribute / Subscript / Starred-non-name fall through silently.


def _rebound_names(tree):
    """Return dict {name: [(reason, lineno), ...]} for every module-
    level name that is REBOUND somewhere inside the module.

    Detection signals (all imply "the name's binding can change at
    runtime"):

    * ``global X`` declaration inside any function body
      (whether or not an Assign follows -- ``global X`` alone
      announces shared mutable state).
    * ``X = <expr>`` inside any function / class body, IF ``X`` is
      ALSO a top-level name (otherwise it's a function-local
      variable that happens to share a name -- not relevant).
      Detection: combine ``ast.Global`` declarations with assign
      targets inside the same function.
    * ``globals()['X'] = <expr>`` at any depth.
    * ``ast.AugAssign`` (``X |= ...``) inside any function whose
      ``ast.Global`` declarations include ``X``.

    The top-level initial assignment does NOT count as a rebind --
    it's the canonical binding the live-forwarding contract
    snapshots.  We need at least one ADDITIONAL rebind site to mark a
    name as mutable.
    """
    top_level = set(_module_level_assigned_names(tree).keys())
    rebinds = {}  # name -> list of (reason, lineno)

    def _record(name, reason, lineno):
        rebinds.setdefault(name, []).append((reason, lineno))

    for node in ast.walk(tree):
        # Pattern: globals()['X'] = ...
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Call)
                        and isinstance(tgt.value.func, ast.Name)
                        and tgt.value.func.id == 'globals'):
                    # The subscript key is the name being rebound.
                    key = tgt.slice
                    # Python 3.9+: slice is the node directly.
                    if isinstance(key, ast.Constant) and isinstance(
                            key.value, str):
                        _record(
                            key.value,
                            'globals()[X] = ... rebind',
                            node.lineno,
                        )

    # For function-scoped rebinds, walk each FunctionDef / AsyncFunctionDef
    # and combine its global decls with its assign targets.
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared_global = set()
        for sub in ast.walk(func):
            if isinstance(sub, ast.Global):
                for nm in sub.names:
                    declared_global.add(nm)
                    # ``global X`` alone signals intent to rebind
                    # shared module state.  Record with the line of
                    # the global statement.
                    if nm in top_level:
                        _record(nm, 'global X declared', sub.lineno)
        # Now find assigns / augassigns inside the function whose
        # target name is in declared_global.
        if not declared_global:
            continue
        for sub in ast.walk(func):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    for nm in _name_targets(tgt):
                        if nm in declared_global and nm in top_level:
                            _record(
                                nm,
                                'X = ... inside global-decl function',
                                sub.lineno,
                            )
            elif isinstance(sub, ast.AugAssign):
                if isinstance(sub.target, ast.Name):
                    nm = sub.target.id
                    if nm in declared_global and nm in top_level:
                        _record(
                            nm, 'AugAssign inside global-decl function',
                            sub.lineno,
                        )
    return rebinds


def _is_private_dunder(name):
    """``__foo__`` style -- skip these (Python machinery, not user
    state).
    """
    return name.startswith('__') and name.endswith('__')


def _mutable_module_globals():
    """Return dict {name: [(reason, lineno), ...]} for every
    mutable module-level global in fft_infra.py.

    "Mutable" = top-level-assigned name that ALSO appears in at
    least one rebind site (``global X``, ``X = ...`` inside a
    function with a matching ``global`` decl, or
    ``globals()['X'] = ...``).
    """
    tree = _parse_fft_infra()
    top_level = _module_level_assigned_names(tree)
    rebinds = _rebound_names(tree)

    out = {}
    for name in top_level:
        if _is_private_dunder(name):
            continue
        if name not in rebinds:
            continue
        out[name] = list(rebinds[name])
    return out


# ---------------------------------------------------------------------------
# Helpers -- discover the lazy backend handles (``cp``, ``pyfftw``)
# whose module-level binding is rebound by ``_ensure_*_loaded()``
# helpers but whose ``global`` decl appears inside a function that
# does not name them via ``X = ...`` at the top level (they ARE
# top-level-assigned to ``None`` at module-import time).
# ---------------------------------------------------------------------------
#
# These are caught by ``_rebound_names`` via the
# "``global X`` declared" branch already; no extra discovery needed.
# Kept as a doc-comment so a future reader doesn't add a redundant
# helper.


# ===========================================================================
# V14.1 -- every mutable global in fft_infra must be in _LIVE_FORWARD_NAMES
# ===========================================================================

def test_v14_every_mutable_fft_infra_global_is_in_live_forward_names():
    """Forward direction: every name in ``fft_infra.py`` that is
    rebound at runtime (setter functions, ``reset_fft_backend``, lazy
    backend loaders, etc.) must be enumerated in
    ``propagation._LIVE_FORWARD_NAMES``.

    A missing entry would let consumer code reading
    ``propagation.X`` after a setter call see the stale import-time
    snapshot instead of the current ``fft_infra.X``.  This is the
    exact class of bug ``AUDIT_V5_1_0_2026_05_20.md`` P3-NEW-F1-1
    flagged for ``_PYFFTW_BAD_SHAPES`` (caught only by manual review,
    not by any walker).  V14 catches the next instance mechanically.
    """
    # Import propagation lazily so an import-time failure surfaces as
    # a test failure rather than a collection error.
    from lumenairy.propagators import propagation

    mutable = _mutable_module_globals()
    whitelist = set(propagation._LIVE_FORWARD_NAMES)
    exempt = set(_V14_FORWARDING_EXEMPTIONS)

    missing = []
    for name, sites in sorted(mutable.items()):
        if name in whitelist:
            continue
        if name in exempt:
            continue
        # Format the rebind-site list for the diagnostic.
        sites_str = ', '.join(
            f'{reason} @ fft_infra.py:{lineno}'
            for reason, lineno in sites
        )
        missing.append(f'  - {name}  ({sites_str})')

    assert not missing, (
        'V14 forward check: fft_infra.py has mutable module-level '
        'globals that are NOT listed in '
        'propagation._LIVE_FORWARD_NAMES.  Consumer code reading '
        '``propagation.X`` after a setter call will see the stale '
        'import-time snapshot rather than the current value.  '
        'Either add each name to ``_LIVE_FORWARD_NAMES`` in '
        'lumenairy/propagators/propagation.py, OR add it to '
        '``_V14_FORWARDING_EXEMPTIONS`` in this file with an inline '
        'comment explaining why the rebind does not require live-'
        'forwarding.\n\nMissing entries:\n' + '\n'.join(missing)
    )


# ===========================================================================
# V14.2 -- every _LIVE_FORWARD_NAMES entry must resolve on propagation
# ===========================================================================

def test_v14_every_live_forward_name_resolves_via_propagation_shell():
    """Reverse direction: every name in
    ``propagation._LIVE_FORWARD_NAMES`` must resolve via
    ``getattr(propagation, name)`` without raising.

    Catches the opposite drift from V14.1: a name removed from
    ``fft_infra.py`` but left in the whitelist would still pass V14.1
    (no mutable global to compare to) yet raise ``AttributeError``
    at consumer call time -- the shell's ``__getattr__`` forwards to
    ``fft_infra`` which no longer defines the name.
    """
    from lumenairy.propagators import fft_infra, propagation

    whitelist = sorted(propagation._LIVE_FORWARD_NAMES)
    broken = []
    for name in whitelist:
        # The forward goes through propagation.__getattr__ -> fft_infra.
        # Use getattr on the shell so we exercise the actual forwarding
        # path, not the fft_infra module attribute directly.
        try:
            getattr(propagation, name)
        except AttributeError as exc:
            broken.append((name, str(exc)))

    assert not broken, (
        'V14 reverse check: propagation._LIVE_FORWARD_NAMES contains '
        'name(s) that no longer resolve via the PEP-562 forwarding '
        'chain.  Either the name was removed from fft_infra.py without '
        'updating the whitelist, or the forwarding stub is broken.  '
        'Each entry below shows the name and the AttributeError '
        'observed when reading ``propagation.<name>``:\n' + '\n'.join(
            f'  - {nm}: {err}' for nm, err in broken
        )
    )


# ===========================================================================
# V14.3 -- mutable-global discovery is non-vacuous
# ===========================================================================

def test_v14_mutable_global_discovery_is_non_vacuous():
    """Sanity check: V14's AST walker must discover at least one
    mutable global in ``fft_infra.py``.  A discovery count of zero
    would mean the walker has degenerated (e.g. the AST traversal
    silently broke after a Python-syntax refactor) and V14.1 would
    pass VACUOUSLY.

    Pinning the minimum at >= 10 keeps the assertion robust to
    future additions / removals of individual globals while still
    catching a degenerated walker.  As of v5.1.1 the walker
    discovers > 15 mutable globals; the floor is set well below
    that.
    """
    mutable = _mutable_module_globals()
    assert len(mutable) >= 10, (
        f'V14 walker discovered only {len(mutable)} mutable global(s) '
        f'in fft_infra.py: {sorted(mutable.keys())!r}.  This is below '
        f'the expected floor (>= 10) and suggests the AST walker has '
        f'silently degenerated -- V14.1 would pass vacuously.  '
        f'Investigate before relaxing this floor.'
    )


# ===========================================================================
# V14.4 -- _PYFFTW_BAD_SHAPES regression pin (audit closure for P3-NEW-F1-1)
# ===========================================================================

def test_v14_pyfftw_bad_shapes_remains_in_whitelist():
    """Explicit regression pin for AUDIT_V5_1_0 P3-NEW-F1-1.

    ``_PYFFTW_BAD_SHAPES`` was the canonical instance of the
    sibling-gap that V14 is designed to retire: ``reset_fft_backend``
    rebinds it via ``_PYFFTW_BAD_SHAPES = set()`` (a fresh object, not
    just ``.clear()``), and pre-v5.1.1 the shell's import-time
    snapshot diverged from ``fft_infra`` after the first reset.  The
    v5.1.1 closure added the name to the whitelist by hand; this pin
    refuses to ship a release that has dropped the entry.
    """
    from lumenairy.propagators import propagation
    assert '_PYFFTW_BAD_SHAPES' in propagation._LIVE_FORWARD_NAMES, (
        '``_PYFFTW_BAD_SHAPES`` was dropped from '
        'propagation._LIVE_FORWARD_NAMES.  This is the canonical '
        'AUDIT_V5_1_0 P3-NEW-F1-1 finding -- the entry MUST stay in '
        'the whitelist (reset_fft_backend() rebinds the global, not '
        'just clears it in-place).  See the comment block in '
        'propagation.py:231-238 for the full rationale.'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
