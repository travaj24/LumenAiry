"""
Process-global knob registry, ``override(...)`` and snapshot/restore.
=====================================================================

The library carries ~20 genuine process-global configuration knobs -- the
FFT dispatch set, the RAM and cache budgets, the storage backend, the
user-library path, the RCWA BLAS cap -- each spelled as a ``set_X`` /
``get_X`` pair on the module that owns it.  The 2026-09-11 audit
(TESTS-ARCH P2-5) measured **53 ``set_`` verbs, 0 context-manager forms
and 0 resets**, in a suite that is deliberately run SERIALLY: a knob a
test forgets to put back silently changes a later test's physics, and the
failure does not reproduce under ``-k`` because the poisoner is not
selected.

This module closes that class with three pieces:

1. :func:`register_knob` -- one line beside each setter, so the knob
   becomes introspectable and restorable.
2. :func:`override` -- a ``with`` form for every registered knob, with
   restore-on-exception and correct nesting.
3. :func:`snapshot` / :func:`restore` -- the pair the test suite's autouse
   fixture uses (``tests/conftest.py``) to make the suite
   order-independent.

Example
-------
>>> import lumenairy as la
>>> with la.override(fft_threads=1, pyfftw_planner='FFTW_ESTIMATE'):
...     ...                                          # doctest: +SKIP
>>> # both knobs are back to their previous values here, in reverse order

Registration is at IMPORT TIME of the module that owns the knob
---------------------------------------------------------------
A knob exists only once its module has been imported, so
:func:`knobs` grows as the library is used::

    >>> import lumenairy                              # doctest: +SKIP
    >>> 'storage_backend' in lumenairy.override.__self__ if False else None
    >>> import lumenairy.io.storage                   # doctest: +SKIP
    >>> # 'storage_backend' is registered from here on

That is deliberate: registering eagerly would mean importing every knob
owner at ``import lumenairy``, which is exactly the import-time cost the
same audit asks us to remove (TESTS-ARCH P2-7).  The consequence to know
is that :func:`snapshot` only covers what is imported at the time it is
called -- see :func:`restore`, which handles the "registered later" case
explicitly.

Scope and threading
-------------------
A knob is PROCESS-global, and so is :func:`override`: it is not a
thread-local scope.  Nesting works (each ``with`` keeps its own saved
values on its own frame, and they unwind in reverse), and calling it from
a worker thread works, but two threads overriding the SAME knob at the
same time race exactly as two threads calling the setter would -- the
one that exits last wins.  Under a ``ThreadPoolExecutor``, set the knob
once around the whole parallel section on the submitting thread rather
than once per worker.  (``elements/rcwa/_core.py::set_blas_threads`` is
the one knob whose REQUEST is thread-local while its APPLICATION is not;
its registration documents that.)

Relationship to ``lumenairy_context``
-------------------------------------
:func:`lumenairy._context.lumenairy_context` predates this module and
scopes a FIXED set of five knobs through named kwargs, with an atexit
restore.  It keeps working unchanged and is the friendlier spelling for
the five it covers; :func:`override` is the generic form that reaches
every registered knob, including ones added later.

Author: Andrew Traverso
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Dict, Iterator, NamedTuple, Tuple

__all__ = [
    'register_knob',
    'override',
    'snapshot',
    'restore',
    'knobs',
    'knob_doc',
]


class _Knob(NamedTuple):
    """One registered process-global knob."""

    name: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    doc: str
    initial: Any


#: name -> _Knob.  Written only by :func:`register_knob` (under
#: :data:`_REGISTRY_LOCK`); read without the lock everywhere else, which is
#: safe because dict reads are atomic under the GIL and registration happens
#: at module-import time.
_REGISTRY: Dict[str, _Knob] = {}
_REGISTRY_LOCK = threading.Lock()


def _same(a: Any, b: Any) -> bool:
    """Best-effort "is this the same knob value?" test.

    Identity first (the common case and the only test that is always
    meaningful), then ``==`` so that benign renormalisations compare equal
    -- ``np.dtype('float64') == np.float64`` is True, and the dtype knobs
    ship as numpy scalar TYPES but come back from their setters as
    ``np.dtype`` INSTANCES.  A comparison that raises or returns a
    non-bool (an array, say) counts as "different", which costs one
    redundant setter call and never skips a needed one.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:       # noqa: BLE001 -- any comparison failure means "differs"
        return False


def register_knob(name: str, *, getter: Callable[[], Any],
                  setter: Callable[[Any], None], doc: str) -> None:
    """Register a process-global configuration knob.

    Call this once, at module scope, immediately below the ``set_X`` /
    ``get_X`` pair it describes -- the knob then becomes visible to
    :func:`override`, :func:`snapshot` and :func:`restore` as soon as that
    module is imported.

    Parameters
    ----------
    name : str
        The knob's identifier, which is also the keyword
        :func:`override` accepts.  By convention it is the setter's name
        with the ``set_`` prefix removed (``set_fft_threads`` ->
        ``'fft_threads'``).  Must be a non-empty string.
    getter : callable
        Zero-argument callable returning the knob's CURRENT value.  It
        must be cheap and SIDE-EFFECT-FREE: the test suite calls it once
        per knob per test.  Where the public accessor has a side effect
        or an expensive fallback (``get_library_path`` creates
        directories; ``get_cache_budget`` queries psutil when no override
        is set), register a module-private accessor that returns the raw
        override instead.
    setter : callable
        One-argument callable that applies a value produced by ``getter``.
        ``setter(getter())`` must be a no-op round trip.
    doc : str
        One line saying what the knob does and what the value means.
        Surfaced by :func:`knob_doc`.

    Raises
    ------
    ValueError
        If ``name`` is not a non-empty string.
    TypeError
        If ``getter`` or ``setter`` is not callable.

    Notes
    -----
    Re-registering the same ``name`` REPLACES the entry (so
    ``importlib.reload`` of an owner module is safe) and re-captures the
    registration-time value used by :func:`restore`.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"register_knob: name must be a non-empty str, got {name!r}.")
    if not callable(getter):
        raise TypeError(
            f"register_knob: getter for knob {name!r} must be callable, got "
            f"{type(getter).__name__}.")
    if not callable(setter):
        raise TypeError(
            f"register_knob: setter for knob {name!r} must be callable, got "
            f"{type(setter).__name__}.")
    initial = getter()
    with _REGISTRY_LOCK:
        _REGISTRY[name] = _Knob(name, getter, setter, str(doc), initial)


def knobs() -> Tuple[str, ...]:
    """Return the names of every knob registered SO FAR, sorted.

    "So far" is load-bearing: a knob appears only once the module that
    owns it has been imported (see the module docstring).
    """
    return tuple(sorted(_REGISTRY))


def knob_doc(name: str) -> str:
    """Return the one-line description supplied at registration.

    Raises
    ------
    ValueError
        If ``name`` is not a registered knob.
    """
    try:
        return _REGISTRY[name].doc
    except KeyError:
        raise ValueError(
            f"knob_doc: unknown knob {name!r}; known: {list(knobs())}."
        ) from None


def snapshot() -> Dict[str, Any]:
    """Return ``{knob_name: current_value}`` for every registered knob.

    A plain dict of the getters' return values -- the values themselves
    are not copied, so a knob whose value is mutable (only
    ``asm_cache_size``, which returns a fresh dict per call) must hand
    back a fresh object from its getter.  Cheap by construction: every
    registered getter is a global read.

    Pairs with :func:`restore`; ``restore(snapshot())`` is a no-op.
    """
    return {name: k.getter() for name, k in _REGISTRY.items()}


def restore(state: Dict[str, Any]) -> None:
    """Put every registered knob back to the value ``state`` records.

    For each registered knob:

    * present in ``state`` -- its setter is called with the recorded
      value **only if the live value differs**;
    * absent from ``state`` -- it was registered after the snapshot was
      taken (its owner module was imported during the interval), so it is
      returned to the value it had at registration time.

    The skip-when-equal rule is not an optimisation detail, it is the
    contract: several setters clear caches as a side effect
    (``set_pyfftw_planner`` drops the pyFFTW plan cache,
    ``set_default_complex_dtype`` drops the ASM H cache,
    ``set_cache_budget`` evicts globally), so an unconditional restore
    after every test would throw away work the tests paid for and would
    make the fixture cost real time.

    Parameters
    ----------
    state : dict
        A mapping produced by :func:`snapshot` (or any subset of it).
        Keys that are not registered knobs are ignored -- a snapshot may
        legitimately outlive a reload.

    Raises
    ------
    TypeError
        If ``state`` is not a mapping.
    """
    if not hasattr(state, 'get') or not hasattr(state, '__contains__'):
        raise TypeError(
            f"restore: state must be a mapping produced by snapshot(), got "
            f"{type(state).__name__}.")
    for name, k in list(_REGISTRY.items()):
        want = state[name] if name in state else k.initial
        if not _same(k.getter(), want):
            k.setter(want)


@contextlib.contextmanager
def override(**knob_values: Any) -> Iterator[None]:
    """Scope one or more process-global knobs to a ``with`` block.

    Every registered knob named as a keyword is set on entry and restored
    on exit -- normal exit, ``return``, ``break`` or exception alike --
    in REVERSE order, so a pair of knobs whose setters interact unwinds
    the way it wound.

    Parameters
    ----------
    **knob_values
        ``knob_name=value`` for any name in :func:`knobs`.  Unknown names
        raise before anything is changed.

    Raises
    ------
    ValueError
        If any name is not a registered knob.  ALL names are validated
        before the first setter runs, so a typo cannot leave half the
        block's knobs applied.  The message lists the known names -- and
        remember that a knob is only registered once its owner module has
        been imported.

    Examples
    --------
    Pin the FFT path for one experiment::

        >>> import lumenairy as la
        >>> with la.override(fft_threads=1, fft_auto_promote=False):
        ...     E = la.angular_spectrum_propagate(E, z, lam, dx)   # doctest: +SKIP

    Nesting composes; each level restores what was live when it entered::

        >>> with la.override(max_ram=8):                # doctest: +SKIP
        ...     with la.override(max_ram=2):
        ...         ...                                  # 2 GB here
        ...     ...                                      # 8 GB here

    Notes
    -----
    * A knob whose requested value already equals the live value is NOT
      re-set (several setters clear caches as a side effect); it is still
      restored correctly, because "restore" is likewise a no-op then.
    * If a setter RAISES partway through entry, the knobs already applied
      are rolled back in reverse before the exception propagates, so the
      process is left exactly as the ``with`` found it.
    * Process-global, not thread-local: see the module docstring.
    """
    unknown = [n for n in knob_values if n not in _REGISTRY]
    if unknown:
        raise ValueError(
            f"override: unknown knob {unknown[0]!r}; known: {list(knobs())}."
            + (f"  (also unknown: {unknown[1:]})" if len(unknown) > 1 else "")
            + "  A knob is registered when the module that owns it is "
              "imported.")

    applied: list = []          # [(knob, previous_value)], entry order
    try:
        for name, value in knob_values.items():
            k = _REGISTRY[name]
            prev = k.getter()
            applied.append((k, prev))
            if not _same(prev, value):
                k.setter(value)
    except BaseException:
        _unwind(applied)
        raise
    try:
        yield
    finally:
        _unwind(applied)


def _unwind(applied) -> None:
    """Restore ``[(knob, previous_value)]`` in reverse entry order.

    A setter that raises on the way out must not strand the knobs that
    have not been restored yet, so each restore is guarded and the FIRST
    failure is re-raised after every other knob has been put back.
    """
    first_exc = None
    for k, prev in reversed(applied):
        try:
            if not _same(k.getter(), prev):
                k.setter(prev)
        except BaseException as exc:      # noqa: BLE001 -- re-raised below
            if first_exc is None:
                first_exc = exc
    applied.clear()
    if first_exc is not None:
        raise first_exc
