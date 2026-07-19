"""
Scoped runtime-environment manager (4.8.1).
============================================

`lumenairy_context` is a context manager that snapshots and restores the
library's process-global runtime settings (default complex dtype, pyFFTW
planner / threads, RAM budget, ASM cache caps) for the duration of a
``with`` block.  Use it to isolate a single experiment's environment from
the rest of a long-running session -- notebooks especially, where global
state set in one cell silently leaks into every cell that follows.

Why a context manager and not per-call kwargs?
----------------------------------------------
Some library state can't be threaded through a per-call kwarg because
it's consumed deep inside library internals the caller never sees -- the
pyFFTW planner flag, the FFT thread count, the ASM kernel cache caps,
the default complex dtype used for internal scratch buffers in
``apply_real_lens_traced``'s parallel-amp path, etc.  These naturally
live as process-global state, and a context manager is the right tool
to scope them.  Per-call kwargs are still the right answer for
operation-level parameters (``N``, ``dx``, ``wavelength``, ``dtype`` on
source factories) -- they describe *what you're doing*, not *how the
library is configured*.

The companion atexit hook (installed at import time by
``lumenairy.__init__``) snapshots the import-time defaults and restores
them on process shutdown so that ``set_default_complex_dtype`` and
friends don't permanently leak state into the next process if the user
calls them outside of any ``with`` block.

Author: Andrew Traverso
"""

from __future__ import annotations

import atexit
import contextlib
from typing import Any, Iterator, Optional, Union

import numpy as np

__all__ = [
    'lumenairy_context',
    'snapshot_globals',
    'apply_globals',
    # v5.2.5 (AUDIT_V5_2_3 P3-F4): the public name ``install_atexit_restore``
    # was renamed to ``_install_atexit_restore`` to signal its private-
    # bootstrap intent (caller is ``lumenairy/__init__.py`` at the end
    # of library import; no user-facing call site).  The legacy name
    # remains importable as a back-compat alias defined at the bottom
    # of this module, but is intentionally NOT in ``__all__``.
    '_install_atexit_restore',
]


def snapshot_globals() -> dict[str, Any]:
    """Return a dict of the library's current global state.

    Captures every knob that :func:`lumenairy_context` can scope:

    - ``complex_dtype``      : :data:`DEFAULT_COMPLEX_DTYPE`
    - ``pyfftw_planner``     : pyFFTW plan-effort flag
    - ``fft_threads``        : pyFFTW thread count (effective)
    - ``max_ram``            : manual RAM override (``None`` = auto)
    - ``asm_cache_size``     : dict from :func:`get_asm_cache_size`

    The returned dict is shaped exactly to round-trip through
    :func:`apply_globals`.  Used internally by :func:`lumenairy_context`
    and by the atexit hook installed at import time.
    """
    from .memory import get_max_ram
    from .propagators.propagation import (
        get_asm_cache_size,
        get_default_complex_dtype,
        get_fft_threads,
        get_pyfftw_planner,
    )

    return {
        'complex_dtype': get_default_complex_dtype(),
        'pyfftw_planner': get_pyfftw_planner(),
        'fft_threads': get_fft_threads(),
        'max_ram': get_max_ram(),
        'asm_cache_size': get_asm_cache_size(),
    }


def apply_globals(state: dict[str, Any]) -> None:
    """Apply a globals snapshot produced by :func:`snapshot_globals`.

    Each field is optional in the dict; missing keys leave the
    corresponding global untouched.  Use as
    ``apply_globals(snapshot_globals())`` for the round-trip identity.

    For each knob the call is **only issued when the requested value
    differs from the current value**.  Several of the underlying
    setters clear caches as a side effect (e.g.
    :func:`set_default_complex_dtype` wipes the ASM H cache;
    :func:`set_pyfftw_planner` wipes the pyFFTW plan cache).  The
    no-op-on-match short-circuit means entering and exiting a
    ``lumenairy_context(complex_dtype=current_value)`` doesn't
    invalidate caches the user already paid to build -- a common
    case when a context manager passes through state it doesn't
    actually want to change.
    """
    from .memory import get_max_ram, set_max_ram
    from .propagators.propagation import (
        get_asm_cache_size,
        get_default_complex_dtype,
        get_fft_threads,
        get_pyfftw_planner,
        set_asm_cache_size,
        set_default_complex_dtype,
        set_fft_threads,
        set_pyfftw_planner,
    )

    if 'complex_dtype' in state:
        new = np.dtype(state['complex_dtype'])
        if np.dtype(get_default_complex_dtype()) != new:
            set_default_complex_dtype(new)
    if 'pyfftw_planner' in state:
        if get_pyfftw_planner() != state['pyfftw_planner']:
            set_pyfftw_planner(state['pyfftw_planner'])
    if 'fft_threads' in state:
        if get_fft_threads() != int(state['fft_threads']):
            set_fft_threads(state['fft_threads'])
    if 'max_ram' in state:
        if get_max_ram() != state['max_ram']:
            set_max_ram(state['max_ram'])
    if 'asm_cache_size' in state:
        if get_asm_cache_size() != state['asm_cache_size']:
            set_asm_cache_size(**state['asm_cache_size'])


@contextlib.contextmanager
def lumenairy_context(
    *,
    complex_dtype: Optional[Any] = None,
    pyfftw_planner: Optional[str] = None,
    fft_threads: Optional[int] = None,
    max_ram: Optional[Union[int, float]] = None,
    asm_cache_max_total_bytes: Optional[int] = None,
    asm_cache_max_bytes_per_entry: Optional[int] = None,
    clear_caches_on_exit: bool = False,
) -> Iterator[None]:
    """Scope a block of code with isolated library runtime settings.

    Every kwarg defaults to ``None`` meaning "leave that knob alone".
    On exit (normal or exception) the prior values are restored.

    Parameters
    ----------
    complex_dtype : numpy dtype, optional
        Library default complex dtype (``np.complex64`` or
        ``np.complex128``).  Controls internal scratch allocations and
        the dtype source factories use when ``dtype=None`` is passed.
        Calling :func:`set_default_complex_dtype` clears the ASM H
        cache, so changing this knob also wipes ``_H_CACHE``.
    pyfftw_planner : str, optional
        One of ``'FFTW_ESTIMATE'`` / ``'FFTW_MEASURE'`` /
        ``'FFTW_PATIENT'`` / ``'FFTW_EXHAUSTIVE'``.  Affects the
        pyFFTW plan effort and clears the existing plan cache.
    fft_threads : int, optional
        Pin pyFFTW + scipy.fft to this thread count.  ``0`` or
        ``None`` restores the affinity-aware default.
    max_ram : int / float, optional
        Manual RAM budget override.  ``< 1024`` is interpreted as
        gigabytes (matching :func:`set_max_ram`); ``>= 1024`` is
        bytes.  Pass ``None`` to leave the existing override alone
        within the ``with`` block (use ``max_ram=0`` if you need to
        explicitly clear an existing override -- handled as bytes,
        which set_max_ram will store as 0 bytes).  Most callers
        either set a positive budget or omit the kwarg.
    asm_cache_max_total_bytes, asm_cache_max_bytes_per_entry : int, optional
        Override the ASM H-cache total / per-entry byte caps for the
        scope of the ``with``.  Other cache parameters (count caps for
        the freq-grid / band-limit caches) are not exposed here -- use
        :func:`set_asm_cache_size` directly if you need to tune those
        and snapshot/restore manually.
    clear_caches_on_exit : bool, default False
        If True, call :func:`clear_asm_caches` on exit to drop the H,
        freq-grid, and band-limit caches.  Useful when the ``with``
        block held large kernel entries that you don't want lingering
        after the experiment finishes; off by default to preserve the
        cache speedup for subsequent calls.

    Examples
    --------
    Run a memory-tight experiment in complex64 with explicit FFT
    tuning, then return to the kernel defaults::

        with la.lumenairy_context(
            complex_dtype=np.complex64,
            pyfftw_planner='FFTW_MEASURE',
            fft_threads=8,
        ):
            E = la.create_gaussian_beam(
                N=4096, dx=2e-6, wavelength=1.55e-6, sigma=1e-3,
                dtype=np.complex64,
            )
            E_out = la.apply_real_lens(
                E, prescription=rx, wavelength=1.55e-6, dx=2e-6)
        # Outside the with: defaults restored automatically.

    Nesting composes the inner context's overrides on top of the
    outer's; on exit each level restores the state that was live
    when it entered::

        with la.lumenairy_context(complex_dtype=np.complex64):
            # complex64 active
            with la.lumenairy_context(pyfftw_planner='FFTW_MEASURE'):
                # complex64 AND FFTW_MEASURE
                ...
            # complex64, FFTW_ESTIMATE (outer was unset)
        # both defaults restored.
    """
    prior = snapshot_globals()

    new_state: dict[str, Any] = {}
    if complex_dtype is not None:
        new_state['complex_dtype'] = complex_dtype
    if pyfftw_planner is not None:
        new_state['pyfftw_planner'] = pyfftw_planner
    if fft_threads is not None:
        new_state['fft_threads'] = fft_threads
    if max_ram is not None:
        new_state['max_ram'] = max_ram
    if (asm_cache_max_total_bytes is not None
            or asm_cache_max_bytes_per_entry is not None):
        cache = dict(prior['asm_cache_size'])
        if asm_cache_max_total_bytes is not None:
            cache['h_max_total_bytes'] = int(asm_cache_max_total_bytes)
        if asm_cache_max_bytes_per_entry is not None:
            cache['h_max_bytes_per_entry'] = int(asm_cache_max_bytes_per_entry)
        new_state['asm_cache_size'] = cache

    # v5.4.6 (audit F-17): apply the new state INSIDE a try so that a
    # setter raising partway through context ENTRY does not leak an
    # already-applied earlier knob.  ``prior`` is snapshotted above
    # (before any mutation), so on an entry-time failure we restore it
    # and re-raise -- leaving process-global state exactly as it was
    # before the ``with`` was attempted.  Pre-v5.4.6 this call sat
    # outside the try/finally, so a mid-apply exception stranded the
    # knobs that had already been set.
    try:
        apply_globals(new_state)
    except BaseException:
        apply_globals(prior)
        raise
    try:
        yield
    finally:
        try:
            apply_globals(prior)
        finally:
            if clear_caches_on_exit:
                # v4.15.0 (P2-CTX-1): a single call to
                # ``clear_asm_caches`` is now the canonical drain
                # entry point.  It routes through the central
                # cache-clearer registry (:mod:`lumenairy._cache_registry`),
                # so every cache-owning module that registered a
                # clearer at import time is drained -- ASM local
                # caches, LG/HG mode-stack, LG polynomial, Zernike
                # basis, through-focus / propagate-system / raytrace
                # JAX, phase-retrieval kernels, berreman / pmm / rcwa
                # / glass / wrapper-merit / eme_jax / bluestein, etc.
                #
                # v5.24.x (audit S4-15): the fallback below no longer
                # hand-lists a subset of clearers.  Pre-fix it enumerated
                # only 7 siblings and OMITTED berreman/pmm/rcwa/glass/
                # wrapper_merit/eme_jax/bluestein -- the exact "fix N,
                # miss N+1" drift the registry was built to retire.  If
                # the ``propagation`` re-export is unavailable (partial
                # install / rename / circular import) we now walk the
                # registry directly via ``clear_all_registered_caches``,
                # which drains EVERY registered clearer rather than a
                # frozen hand-list.  This preserves the v4.13.1 P1-E
                # defence-in-depth guarantee (a broken ``propagation``
                # link does not strand the rest of the cache hygiene)
                # without the maintenance hazard of a divergent subset.
                _chain_called = False
                try:
                    from .propagators.propagation import clear_asm_caches
                    clear_asm_caches()
                    _chain_called = True
                except (ImportError, RuntimeError, AttributeError):
                    pass
                if not _chain_called:
                    # v5.24.x (audit S4-15) defence-in-depth fallback:
                    # canonical chain path unavailable.  Walk the
                    # central registry directly -- it drains every
                    # registered clearer, so no cache is silently
                    # skipped.  The registry lives in the package root
                    # with no propagation-layer dependency, so it is
                    # importable even when ``propagation`` is not.
                    try:
                        from ._cache_registry import (
                            clear_all_registered_caches,
                        )
                        clear_all_registered_caches()
                    except (ImportError, RuntimeError, AttributeError):
                        pass


def _atexit_restore(snapshot: dict[str, Any]) -> None:
    """Restore a globals snapshot at process exit.

    Designed to swallow exceptions silently: an atexit handler that
    raises just emits noise to stderr and accomplishes nothing.  The
    process is going away anyway; the goal of the hook is to leave
    the file system / external state in a clean shape, not to enforce
    library invariants on the way out.
    """
    # KEEP-AS-IS broad-except: atexit handlers must NEVER let an
    # exception propagate -- the module-level globals (including
    # ``apply_globals`` itself) may already be torn down by the
    # interpreter shutdown sequence, so even AttributeError /
    # NameError needs to be tolerated here.
    try:
        apply_globals(snapshot)
    except Exception:
        pass


_ATEXIT_INSTALLED = False
_ATEXIT_SNAPSHOT: Optional[dict[str, Any]] = None


def _install_atexit_restore() -> None:
    """Register an atexit handler that restores import-time defaults.

    Private bootstrap helper.  v5.2.5 (AUDIT_V5_2_3 P3-F4): renamed
    from ``install_atexit_restore`` to the underscore-prefixed
    form; the helper is called exactly once at the end of
    :mod:`lumenairy.__init__` during library import and has no
    user-facing call site.  ``install_atexit_restore`` (the legacy
    public name) is preserved as a back-compat alias below in this
    module for any external caller that imported it directly.

    Idempotent: subsequent calls are no-ops.  Called once at the end
    of :mod:`lumenairy.__init__` so that any user code that calls
    ``set_default_complex_dtype`` / ``set_pyfftw_planner`` etc. at
    module scope still leaves the process exit with library defaults
    intact.  Mostly a paranoia hook -- relevant if a long-running
    test harness or notebook server invokes user code that sets
    library defaults and then expects subsequent unrelated runs in
    the same process to see the original defaults.
    """
    global _ATEXIT_INSTALLED, _ATEXIT_SNAPSHOT
    if _ATEXIT_INSTALLED:
        return
    _ATEXIT_SNAPSHOT = snapshot_globals()
    atexit.register(_atexit_restore, _ATEXIT_SNAPSHOT)
    _ATEXIT_INSTALLED = True


# v5.2.5 (AUDIT_V5_2_3 P3-F4): legacy public name preserved as a
# back-compat alias.  Pre-v5.2.5 the function was named
# ``install_atexit_restore`` (no leading underscore) which made it
# look like a user-facing API surface despite being a private
# bootstrap helper.  External callers that imported it by the old
# name continue to work; new code should use the underscore-
# prefixed canonical name.
install_atexit_restore = _install_atexit_restore
