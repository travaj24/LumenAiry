"""Shared deprecation-warning utilities for 4.7+ API-consistency work.

The 4.7 polish pass adopted the standard Python deprecation pattern --
warn for one minor release, change defaults next, remove later --
across the API-consistency items called out in
``lumenairy_4_6_polish_pass.md``.  This module centralises the
mechanics so each warning has a uniform message format and the
warnings can all be silenced with a single ``warnings.simplefilter``
incantation in user code.

v5.2 (ROADMAP opportunistic item -- "_deprecation.py orphan helpers"):
``warn_deprecated_kwarg``, ``warn_renamed_function``, and
``warn_deprecated_default`` are not currently called by any internal
site.  They remain exported (and exercised by the test suite via the
``_NO_DEFAULT`` sentinel + the deprecated_alias decorator) because:
(a) deletion would silently break any external caller importing them
by name -- we have no telemetry on out-of-repo use; (b) they document
the canonical message format for future deprecation cycles, so
keeping them avoids re-inventing the contract.  If a future v5.x
deprecation lands without using these helpers, that is itself a
sibling-gap pattern flagged by audit cadence.

The library raises ``DeprecationWarning`` (the standard since PEP 565
restored the default-visible behaviour for ``__main__``).  Callers
who want to suppress them temporarily during migration can do::

    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning,
                            module=r'lumenairy.*')

or, to surface them as errors in CI::

    warnings.filterwarnings('error', category=DeprecationWarning,
                            module=r'lumenairy.*')
"""
from __future__ import annotations

import functools
import sys
import warnings
from typing import Any, Callable, Optional

__all__ = [
    'warn_deprecated_kwarg',
    'warn_deprecated_alias',
    'deprecated_alias',
    'warn_renamed_function',
    'warn_deprecated_default',
    'warn_deprecated_signature',
    # v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A
    # "deprecation registry rot"): the removal-schedule registry.
    'NEXT_REMOVAL_VERSION',
    'REMOVAL_SCHEDULE',
    'API_TRANSITION_VERSION',
    'resolve_removal_version',
    'check_removal_schedule',
    # v4.15.1 (Agent E): pickle-safe sentinel helpers; the unpickler
    # must be importable by name at the module top level for the
    # ``_Sentinel.__reduce__`` protocol to round-trip cleanly.
    '_sentinel_unpickle',
]


# ===========================================================================
# Removal-schedule registry
# ===========================================================================
# v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A
# "deprecation registry rot").  MEASURED defect: the removed-in banner
# emitted ``will be removed in v5.27`` from a v5.29.0 library -- i.e. the
# message advertised a horizon the release had already blown through.  Ten
# of twelve live deprecations were past their stated removal version (eight
# said v5.0).
#
# The bug is structural, not a typo: the four message builders below
# interpolated ``version_removed`` verbatim, so NOTHING in the library ever
# compared a stated horizon against the running ``__version__``.  Every
# call site was free to rot independently, and CI could not see it (the
# pins assert the version STRING appears, which a stale string does).
#
# The fix keeps the mechanics in exactly one place:
#
#   * :data:`REMOVAL_SCHEDULE` is the registry of re-scheduled horizons --
#     ``{stated at the shim site: live removal version}``.  Re-scheduling
#     an overdue deprecation is a one-line edit HERE; the shim itself is
#     never removed by this mechanism (removal is a release decision for
#     the module owner).
#   * :func:`resolve_removal_version` maps any stated horizon onto the live
#     one, and -- as a backstop for a site that rots without a registry
#     entry -- promotes ANY already-shipped horizon to
#     :data:`NEXT_REMOVAL_VERSION`.  A banner therefore cannot advertise a
#     removal version <= ``lumenairy.__version__`` again, whatever the call
#     site says.
#   * The emitted text names the live horizon and, when they differ, the
#     original one (``will be removed in v5.32 (rescheduled from v5.27)``)
#     so a caller reading the warning can see the slip rather than a
#     silently-moved goalpost.
#   * :func:`check_removal_schedule` makes the registry self-checking; the
#     pin in ``tests/unit/test_niche_audit_w3_ui_deprecation.py`` asserts it
#     returns no violations, so a shipped release cannot carry a horizon it
#     has already passed.

#: Removal horizon for deprecations whose stated version has shipped.  Set
#: it to a version the project can realistically hit; bumping it is a
#: deliberate one-line slip, recorded in the CHANGELOG.
NEXT_REMOVAL_VERSION = '5.32'

#: Re-scheduled horizons: ``{version as written at the shim call site:
#: live removal version}``.  Keys are the ORIGINAL (now shipped) schedule
#: so the message can name both; values must lie in the future.
#:
#: ``'5.27'`` -- the v5.25 ``seed=`` -> ``rng=`` and ``sigma=`` -> ``w0=``
#: source-factory kwarg deprecations (``sources/core.py``'s
#: ``_DEPRECATION_VERSION_REMOVED``).  Two releases were budgeted; the
#: shims are still shipping at v5.29 and are NOT removed here.
REMOVAL_SCHEDULE: dict[str, str] = {
    '5.27': NEXT_REMOVAL_VERSION,
}

#: Version at which the deferred **API-contract transitions** land -- the
#: default-flip counterpart to :data:`NEXT_REMOVAL_VERSION` (which schedules
#: shim *removals*).  A transition changes a default value rather than
#: deleting a name, so it needs its own registry entry: nothing is removed at
#: this version and the legacy behaviour stays reachable behind an explicit
#: argument.
#:
#: v5.30 (roadmap ``docs/roadmap_deferred_2026_07_21.md`` Part F1, audit P5 --
#: owner decision).  Scheduled here:
#:
#: * :func:`lumenairy.propagators.dispatch.propagate` -- the DEFAULT return
#:   becomes a :class:`~lumenairy.propagators.PropagationResult` for every
#:   method (roadmap F1 option 4, the option costed as least-breaking).  From
#:   v5.30 a ``DeprecationWarning`` fires whenever the default path hands back
#:   the unstable legacy contract (bare ndarray **or** ``(E, dx_out, dy_out)``
#:   triple); ``return_result=True`` (stable) and ``return_result=False``
#:   (legacy shapes, kept available past the flip) are both silent.
#:
#: NOT scheduled here (decided against in the same pass):
#:
#: * ``PropagationResult.__iter__`` -- stays **2-item** ``(field,
#:   intermediates)`` permanently (audit P16).  Option 4 keeps
#:   ``return_result=False`` available, so 3-tuple unpackers migrate by
#:   naming the legacy contract instead of by us re-arity-ing iteration --
#:   which would break the ``E, inter = propagate_through_system(...,
#:   return_result=True)`` callers that the 2-item form exists for.
#:
#: Bound to :data:`NEXT_REMOVAL_VERSION` by construction, so
#: :func:`check_removal_schedule`'s "lies in the future" invariant covers it
#: too and a shipped release cannot advertise a transition it has passed.
API_TRANSITION_VERSION = NEXT_REMOVAL_VERSION


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a ``'5.29.0'`` / ``'v5.27'`` / ``'4.15.1'`` version string
    into a comparable 3-tuple, tolerating suffixes (``'5.30.0rc1'``).

    Missing components read as 0, so ``'5.27' -> (5, 27, 0)`` compares
    correctly against ``'5.29.0' -> (5, 29, 0)``.  Unparseable chunks
    read as 0 rather than raising: a malformed version must not turn a
    deprecation warning into an exception.
    """
    parts: list[int] = []
    for chunk in str(version).strip().lstrip('vV').split('.'):
        digits = ''
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _current_version() -> str:
    """Return the running ``lumenairy.__version__``, or ``''`` if unknown.

    Read from :data:`sys.modules` rather than by importing the package:
    ``lumenairy/__init__.py`` imports THIS module (for
    :func:`deprecated_alias`) long before it binds ``__version__``, so an
    import here would either cycle or read a half-initialised module.  At
    warning time the package is fully loaded, so the lookup succeeds; the
    ``''`` fallback (partial init, or ``_deprecation`` imported
    standalone) simply disables the overdue backstop -- the explicit
    :data:`REMOVAL_SCHEDULE` entries still apply.
    """
    mod = sys.modules.get('lumenairy')
    return str(getattr(mod, '__version__', '') or '')


def resolve_removal_version(
    version_removed: Optional[str],
) -> Optional[str]:
    """Map a stated removal version onto the live one.

    Parameters
    ----------
    version_removed : str or None
        Removal version as written at the deprecation call site.

    Returns
    -------
    str or None
        ``None`` when ``version_removed`` is ``None`` (the message then
        states no horizon at all).  Otherwise the live horizon: the
        :data:`REMOVAL_SCHEDULE` entry if
        one exists, else :data:`NEXT_REMOVAL_VERSION` when the stated
        version has already shipped, else the stated version unchanged.
    """
    if version_removed is None:
        return None
    stated = str(version_removed).strip().lstrip('vV')
    live = REMOVAL_SCHEDULE.get(stated, stated)
    current = _current_version()
    if current and _version_tuple(live) <= _version_tuple(current):
        # Backstop: a site that rotted without a registry entry.
        return NEXT_REMOVAL_VERSION
    return live


def _format_removal(version_removed: Optional[str], *,
                    verb: str = 'removed') -> str:
    """Build the ``, will be removed in vX`` clause of a warning message.

    Single source of the removed-in banner for all four message builders
    below (pre-v5.30 each interpolated ``version_removed`` itself, which
    is how four independent copies of the same rot survived).  ``verb`` is
    ``'removed'`` for shims and ``'required'`` for deprecated defaults.
    """
    if not version_removed:
        return ''
    stated = str(version_removed).strip().lstrip('vV')
    live = resolve_removal_version(stated)
    if live == stated:
        return f', will be {verb} in v{live}'
    return f', will be {verb} in v{live} (rescheduled from v{stated})'


def check_removal_schedule() -> list[str]:
    """Return a list of registry inconsistencies; empty == self-consistent.

    Checked invariants:

    1. :data:`NEXT_REMOVAL_VERSION` lies in the future.
    2. Every re-scheduled horizon (a :data:`REMOVAL_SCHEDULE` value) lies
       in the future.
    3. Every key is a horizon that HAS shipped -- a key that is still in
       the future would silently move a live deprecation's goalpost
       instead of documenting a slip.
    4. :func:`resolve_removal_version` returns a future version for every
       key and value.

    Returns human-readable strings so a failing pin names the offender.
    """
    current = _current_version()
    problems: list[str] = []
    if not current:
        return ['lumenairy.__version__ is not importable; cannot check '
                'the removal schedule']
    cur_t = _version_tuple(current)
    if _version_tuple(NEXT_REMOVAL_VERSION) <= cur_t:
        problems.append(
            f"NEXT_REMOVAL_VERSION={NEXT_REMOVAL_VERSION!r} is not after "
            f"the running version {current!r}")
    for stated, live in REMOVAL_SCHEDULE.items():
        if _version_tuple(live) <= cur_t:
            problems.append(
                f"REMOVAL_SCHEDULE[{stated!r}]={live!r} is not after the "
                f"running version {current!r}")
        if _version_tuple(stated) > cur_t:
            problems.append(
                f"REMOVAL_SCHEDULE[{stated!r}] re-schedules a horizon that "
                f"has NOT shipped yet (running {current!r}); remove the "
                f"entry or fix the call site instead")
        for probe in (stated, live):
            resolved = resolve_removal_version(probe)
            if resolved is None or _version_tuple(resolved) <= cur_t:
                problems.append(
                    f"resolve_removal_version({probe!r}) -> {resolved!r} is "
                    f"not after the running version {current!r}")
    return problems


class _Sentinel:
    """Distinct singleton used as a default-argument marker.

    ``arg=_NO_DEFAULT`` lets callers distinguish "didn't pass a value"
    from "passed None" -- needed to warn-on-default-use without
    breaking explicit ``None`` callers.

    v4.15.1 (Agent E): pickle-safe singleton via a name-keyed registry
    + ``__reduce__``.  Subclasses register themselves on instantiation
    and unpickle through :func:`_sentinel_unpickle` so the result is
    ``is``-identical to the registry singleton (rather than a fresh
    instance).  This is the canonical Python pattern for singleton
    sentinels that cross pickle boundaries (e.g. distributed merit
    evaluation, multiprocessing workers, joblib caches).
    """
    __slots__ = ('_name',)

    def __init__(self, name: str) -> None:
        self._name = name
        # Register this instance as the canonical sentinel for its
        # name.  Re-registration is allowed (idempotent on the same
        # name -- last writer wins) so module reloads / test isolation
        # don't break the invariant.  The registry survives pickle
        # round-trips because :func:`_sentinel_unpickle` reads from
        # it.
        _SENTINEL_REGISTRY[name] = self

    def __repr__(self) -> str:
        return f'<{self._name}>'

    def __bool__(self) -> bool:  # noqa: D401 — sentinel is always falsy
        return False

    def __reduce__(self) -> tuple[Callable[[str], '_Sentinel'], tuple[str]]:
        """Pickle as a name lookup so unpickling returns the singleton.

        Returns the tuple ``(_sentinel_unpickle, (self._name,))`` --
        the standard ``copyreg``-style reconstructor protocol.  When
        the pickle is loaded, Python calls
        ``_sentinel_unpickle(self._name)`` which looks up the existing
        instance in :data:`_SENTINEL_REGISTRY` instead of creating a
        new one.  Result: ``pickle.loads(pickle.dumps(x)) is x`` holds
        for every ``_Sentinel`` subclass instance.
        """
        return (_sentinel_unpickle, (self._name,))


# v4.15.1 (Agent E): name-keyed registry of every ``_Sentinel`` instance
# ever constructed.  Used by :func:`_sentinel_unpickle` to return the
# pre-existing singleton on unpickle rather than constructing a fresh
# instance (which would break ``is`` identity).  Module-level (not
# class-level) so subclasses share the same registry.
_SENTINEL_REGISTRY: dict[str, '_Sentinel'] = {}


def _sentinel_unpickle(name: str) -> '_Sentinel':
    """Return the singleton ``_Sentinel`` registered under ``name``.

    Used as the reconstructor target of ``_Sentinel.__reduce__``.  If
    the registry lookup fails (e.g. the sentinel's defining module was
    not imported on the receiving side) we raise :class:`ImportError`
    with an actionable message rather than silently constructing a
    fresh base :class:`_Sentinel`.  The pre-v4.15.2 fallback path
    produced a *base* ``_Sentinel`` that compared ``False`` under
    ``isinstance`` checks against the original subclass (e.g.
    ``_ZeroApertureMaskSentinel``), silently downgrading caller
    semantics on receivers where the subclass-defining module had not
    yet been imported.  The audit (AUDIT_V4_15_1, P2) flagged this as
    a latent bug in distributed pipelines with delayed imports
    (joblib workers, dask distributed, multiprocessing Pool workers
    that ``cloudpickle`` a callable referencing the sentinel before
    the worker has imported the module).  Strict raise surfaces the
    timing issue at the unpickle site instead of letting the silent
    downgrade propagate downstream.
    """
    inst = _SENTINEL_REGISTRY.get(name)
    if inst is not None:
        return inst
    raise ImportError(
        f"_sentinel_unpickle: no _Sentinel registered under "
        f"name={name!r}.  The defining module has likely not been "
        f"imported on the receiving side.  To unpickle a "
        f"{name!r}-class sentinel, import its defining module "
        f"first (e.g. ``import lumenairy.optimize.core`` for "
        f"``_ZERO_APERTURE_MASK``, ``import lumenairy.elements."
        f"polarization`` for ``_ANGLE_UNSET``, ``import lumenairy."
        f"_deprecation`` for ``NO_DEFAULT``).  See AUDIT_V4_15_1 P2 "
        f"closure for the v4.15.2 strict-raise rationale."
    )


class _NoDefaultSentinel(_Sentinel):
    """Singleton sentinel for "argument was not explicitly passed".

    v4.15.2 (Agent E, P3): dedicated subclass for consistency with
    :class:`_ZeroApertureMaskSentinel` and :class:`_AngleUnsetSentinel`.
    Pre-v4.15.2 ``_NO_DEFAULT`` was a bare ``_Sentinel('NO_DEFAULT')``
    instance, which differed cosmetically from the other two sentinels.
    No behaviour change: the new subclass overrides nothing and the
    singleton instance is still keyed by the ``'NO_DEFAULT'`` registry
    name.
    """
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__('NO_DEFAULT')


_NO_DEFAULT = _NoDefaultSentinel()


def _emit(msg: str, *, stacklevel: int = 3) -> None:
    """Emit a ``DeprecationWarning`` with consistent stacklevel.

    ``stacklevel=3`` lets the warning point at the *caller* of the
    public function rather than the body of the deprecation helper.
    """
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def warn_deprecated_kwarg(
    old_name: str,
    new_name: str,
    *,
    function: str,
    version_added: str = '4.7',
    version_removed: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Warn that a keyword argument was renamed.

    Parameters
    ----------
    old_name : str
        Deprecated kwarg name.
    new_name : str
        The replacement kwarg name.
    function : str
        Fully qualified name of the public function (for the message).
    version_added : str
        Version in which the deprecation began.
    version_removed : str, optional
        Version in which removal is scheduled (only stated if known).
        Routed through :func:`resolve_removal_version`, so an already-
        shipped horizon is reported as the live one.
    stacklevel : int
        Passed through to ``warnings.warn``.
    """
    removal = _format_removal(version_removed)
    _emit(
        f"{function}: keyword argument '{old_name}' is deprecated since "
        f"v{version_added}{removal}; use '{new_name}' instead.",
        stacklevel=stacklevel,
    )


def warn_deprecated_alias(
    old_name: str,
    new_name: str,
    *,
    version_added: str = '4.7',
    version_removed: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Warn that a top-level function alias has been renamed."""
    removal = _format_removal(version_removed)
    _emit(
        f"{old_name}() is a deprecated alias since v{version_added}"
        f"{removal}; use {new_name}() instead.",
        stacklevel=stacklevel,
    )


def deprecated_alias(
    new_func: Callable[..., Any],
    *,
    old_name: str,
    version_added: str = '4.7',
    version_removed: Optional[str] = None,
) -> Callable[..., Any]:
    """Return a thin wrapper that calls ``new_func`` after emitting a
    rename warning.

    Useful for exposing back-compat names at module top level::

        old_function = deprecated_alias(new_function,
                                         old_name='old_function')
    """
    new_name = getattr(new_func, '__qualname__', new_func.__name__)

    @functools.wraps(new_func)
    def _shim(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated_alias(
            old_name, new_name,
            version_added=version_added,
            version_removed=version_removed,
        )
        return new_func(*args, **kwargs)

    # Make introspection accurate: the shim advertises the old name
    # and its docstring carries the deprecation note.
    _shim.__name__ = old_name
    _shim.__doc__ = (
        f"Deprecated alias for :func:`{new_name}` (since v{version_added}).\n"
        f"\n{new_func.__doc__ or ''}"
    )
    return _shim


def warn_renamed_function(
    old_name: str,
    new_name: str,
    *,
    version_added: str = '4.7',
    stacklevel: int = 3,
) -> None:
    """Equivalent to :func:`warn_deprecated_alias` but more explicit
    when the call site is the renamed function itself."""
    warn_deprecated_alias(
        old_name, new_name,
        version_added=version_added,
        stacklevel=stacklevel,
    )


def warn_deprecated_default(
    arg_name: str,
    default_value: Any,
    *,
    function: str,
    version_added: str = '4.7',
    version_removed: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Warn that an argument's default value is deprecated and the
    argument will be required in a future release.

    Typical use::

        from .._deprecation import _NO_DEFAULT, warn_deprecated_default

        def keplerian_telescope(f_obj, f_eye, *, wavelength=_NO_DEFAULT):
            if wavelength is _NO_DEFAULT:
                warn_deprecated_default(
                    'wavelength', 550e-9, function='keplerian_telescope',
                    version_removed='5.32',
                )
                wavelength = 550e-9
            ...
    """
    removal = _format_removal(version_removed, verb='required')
    _emit(
        f"{function}: relying on the default value of '{arg_name}' "
        f"({default_value!r}) is deprecated since v{version_added}"
        f"{removal}; pass it explicitly.",
        stacklevel=stacklevel,
    )


def warn_deprecated_signature(
    *,
    function: str,
    old_signature: str,
    new_signature: str,
    version_added: str = '4.15',
    version_removed: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Warn that a legacy positional call form is deprecated.

    Used when a function has been re-shaped (e.g. legacy positional
    ``f(size, N, dx, wavelength)`` -> canonical kwarg-only
    ``f(*, N, dx, wavelength, size)``) and the old call form is still
    accepted with a back-compat shim.

    Parameters
    ----------
    function : str
        Fully qualified name of the function being called.
    old_signature : str
        Human-readable representation of the deprecated call form
        (e.g. ``"Source.gaussian(w0, N, dx, wavelength)"``).
    new_signature : str
        Human-readable representation of the canonical call form
        (e.g. ``"Source.gaussian(*, N, dx, wavelength, w0)"``).
    version_added : str
        Version in which this deprecation began.
    version_removed : str, optional
        Version in which removal is scheduled (only stated if known).
        Routed through :func:`resolve_removal_version`, so an already-
        shipped horizon is reported as the live one.
    stacklevel : int
        Passed through to ``warnings.warn``.
    """
    removal = _format_removal(version_removed)
    _emit(
        f"{function}: legacy positional call form "
        f"``{old_signature}`` is deprecated since v{version_added}"
        f"{removal}; use the canonical form ``{new_signature}`` instead.",
        stacklevel=stacklevel,
    )
