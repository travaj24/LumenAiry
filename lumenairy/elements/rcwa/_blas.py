"""RCWA BLAS-thread controls -- the optional, opt-in cap on the BLAS pool
used by the NumPy / CuPy RCWA and PMM solves.

A genuine LEAF: it imports the standard library, ``lumenairy._knobs`` (itself a
leaf) and nothing else from the package, so every engine can depend on it
without an edge in the other direction.  ``_core`` re-exports the seven names
its ``__all__`` publishes (``_BLAS_STATE``, ``_get_blas_threads``,
``set_blas_threads``, ``rcwa_blas_threads``, ``_blas_threads_quiet``,
``_blas_limit``, ``_with_blas_limit``) so every existing import path resolves
unchanged and to the SAME objects.

WHERE TO MONKEYPATCH THE STATE.  The five module-level mutable names below --
``_BLAS_STATE``, ``_BLAS_WARNED_UNCONTROLLABLE``, ``_BLAS_CONTROLLER``,
``_BLAS_CONTROLLER_UNAVAILABLE``, ``_BLAS_CONTROLLER_LOCK`` -- and the three
functions that read them at call time (``_threadpoolctl_available``,
``_warn_blas_uncontrollable``, ``_get_blas_controller``) are read through THIS
module's globals, so a test that substitutes one must set it HERE, not on
``_core``.  They are deliberately NOT re-exported into ``_core``, and ``_core``
carries no PEP 562 forward for them, so a stale ``setattr(_core, ...)`` raises
``AttributeError`` loudly instead of binding a shadow attribute nothing reads.
``_BLAS_STATE`` is the one exception -- it is in ``_core.__all__``, it is
re-exported, and because it is a ``threading.local()`` OBJECT that every reader
mutates through (never rebinds), the re-export names the same object and a
mutation through either path is seen by both.
"""
from __future__ import annotations

import contextlib
import functools
import threading
import warnings
from typing import Optional

from ..._knobs import register_knob as _register_knob

# Optional BLAS-thread cap for the NumPy/CuPy solve.  The dense non-Hermitian
# eigensolver (LAPACK zgeev, largely serial) plus the S-matrix BLAS3 thrash
# under thread oversubscription on many-core boxes, so capping the BLAS pool
# to a few threads is a MODEST, machine-dependent ~2-3x speedup at moderate N
# with ZERO numerics change.  Opt-in (None = leave the environment's threading
# untouched) because the optimum is configuration-dependent and a global
# thread change shouldn't be forced on the caller.
#
# WHAT IS AND IS NOT THREAD-LOCAL (corrected, M4 2026-08-04).  The REQUESTED
# cap below is thread-local, so two threads can ASK for different caps without
# overwriting each other's request.  APPLYING it is NOT: :func:`_blas_limit`
# goes through ``threadpoolctl``, and on OpenBLAS that calls
# ``openblas_set_num_threads()``, which is PROCESS-GLOBAL.  MEASURED on
# Windows/OpenBLAS 0.3.31, 24 threads: a worker thread entering
# ``threadpool_limits(1)`` takes the MAIN thread's reported pool to 1 as well,
# and the worker's exit restores 24 for everyone -- including siblings that are
# still inside a solve.
#
# CONSEQUENCE, and the rule that follows from it: N concurrent
# enter/exit pairs on one process-global setting RACE, and a solve whose
# BLAS thread count changes underneath it returns different last bits (a
# different GEMM/LAPACK reduction order).  So a caller that needs
# reproducible results across worker counts must apply the cap ONCE, around
# the whole parallel section, on the calling thread -- never once per worker.
# :meth:`RCWAStack.solve_vs_wavelength` does exactly that; see the comment at
# its dispatch.  The prior text here claimed the save/restore was thread-local
# and therefore race-free; it is not, and that is what broke the sweep's
# byte-identity pin (a few ULP in T, ~50-70% of runs, only when
# ``threadpoolctl`` is installed AND the environment pool is > 1).
_BLAS_STATE = threading.local()


def _get_blas_threads() -> Optional[int]:
    return getattr(_BLAS_STATE, "n", None)


# ``threadpoolctl`` is an OPTIONAL dependency and is NOT bundled with numpy
# (audit M6 2026-07-25 -- two in-code comments claimed otherwise).  Without it
# the cap cannot be applied at all, so a requested cap is inert; warn once
# rather than let the caller believe a reported cap is in force.
_BLAS_WARNED_UNCONTROLLABLE = False


def _threadpoolctl_available() -> bool:
    """True when ``threadpoolctl`` is importable -- i.e. when a requested BLAS
    cap can actually be APPLIED (via ``ThreadpoolController`` or the legacy
    ``threadpool_limits``)."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        return False
    return True


def _warn_blas_uncontrollable() -> None:
    """Warn ONCE per process when a BLAS cap is requested with no controller
    installed, so the cap is silently inert (audit M6 2026-07-25)."""
    global _BLAS_WARNED_UNCONTROLLABLE
    if _BLAS_WARNED_UNCONTROLLABLE or _threadpoolctl_available():
        return
    _BLAS_WARNED_UNCONTROLLABLE = True
    warnings.warn(
        "rcwa: BLAS-THREAD CAP IS INERT -- set_blas_threads(...) / "
        "rcwa_blas_threads(...) / the @_with_blas_limit wrapper on every public "
        "RCWA entry point all need the `threadpoolctl` package, which is NOT "
        "installed.  The solve runs at the environment's default threading "
        "even though _get_blas_threads() keeps reporting the requested value.  "
        "THIS IS NOT A MICRO-OPTIMISATION: on an oversubscribed many-core box "
        "the default pool is catastrophically slow for these small dense "
        "eigen/inverse kernels -- MEASURED on a 24-thread Windows OpenBLAS "
        "0.3.31 build, inv() of a 163x163 complex matrix takes 2.29 s unpinned "
        "against 0.0057 s pinned to one thread (400x), and a 1-D TM solve at "
        "n_orders=81 takes 18.2 s instead of 0.13 s (140x).  Fix it either way: "
        "`pip install threadpoolctl` (tiny, pure Python) so this library's own "
        "cap works, or set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / "
        "MKL_NUM_THREADS=1 in the environment BEFORE importing numpy.",
        stacklevel=3)



def set_blas_threads(n: Optional[int]) -> None:
    """Cap the BLAS thread pool used by subsequent NumPy/CuPy RCWA solves on
    the CURRENT thread.

    On a thread-oversubscribed many-core box the dense ``zgeev`` eigensolver
    (largely serial) and the S-matrix BLAS3 contend, so a small cap (the
    measured optimum is ~2) gives a modest ~2-3x speedup at moderate truncation
    -- machine-dependent, with no change to the numbers.  Pass ``None`` to
    restore the default (untouched) threading.  Has no effect on the JAX path
    (XLA manages its own threads).  For a scoped cap use
    :func:`rcwa_blas_threads`.

    The REQUEST recorded here is thread-local; APPLYING it is not.  On OpenBLAS
    the underlying ``threadpoolctl`` call is process-global (MEASURED, M4
    2026-08-04 -- see the ``_BLAS_STATE`` comment above), so two threads that
    hold DIFFERENT caps at the same time do interfere: whichever exits first
    restores the pool for both.  Set one cap around a parallel section rather
    than one cap per worker.

    REQUIRES ``threadpoolctl``.  Without it the cap cannot be applied and this
    call is INERT -- it then warns ONCE per process (audit M6 2026-07-25: the
    cap was silently ignored while :func:`_get_blas_threads` kept reporting it,
    so a caller measuring "no speed-up" had no way to see why).
    """
    _BLAS_STATE.n = None if n is None else max(1, int(n))
    if _BLAS_STATE.n is not None:
        _warn_blas_uncontrollable()



# The only knob in the library whose REQUEST is thread-local (see the
# ``_BLAS_STATE`` comment above): registering it makes the CALLING thread's
# request snapshot/restorable, which is what the suite needs -- a test that
# calls ``set_blas_threads(2)`` on the main thread and forgets to put it back
# otherwise changes every later solve in the process.  ``lumenairy.override(
# blas_threads=...)`` therefore scopes the caller's thread, exactly like the
# pre-existing :func:`rcwa_blas_threads`, which stays the local spelling.
_register_knob(
    'blas_threads',
    getter=_get_blas_threads, setter=set_blas_threads,
    doc="BLAS thread cap requested for RCWA solves on the CURRENT thread; "
        "None (shipped) leaves the environment's threading untouched.  "
        "Requires threadpoolctl to have any effect.")


@contextlib.contextmanager
def rcwa_blas_threads(n: Optional[int]):
    """Context manager that caps the BLAS pool for RCWA solves within the
    ``with`` block on the current thread (see :func:`set_blas_threads`, whose
    ``threadpoolctl`` requirement and once-per-process inert-cap warning this
    shares); restores the prior setting on exit."""
    prev = _get_blas_threads()
    set_blas_threads(n)
    try:
        yield
    finally:
        _BLAS_STATE.n = prev



@contextlib.contextmanager
def _blas_threads_quiet(n: Optional[int]):
    """:func:`rcwa_blas_threads` without the inert-cap warning -- for the
    LIBRARY's own caps around threaded sweeps (audit M6): the user did not
    request those, so surfacing "your cap is inert" there would turn a
    diagnostic into noise on an ordinary sweep.  The public setters keep the
    warning.

    Use it AROUND a parallel section, on the calling thread -- not inside each
    worker.  Applying a cap is process-global on OpenBLAS, so per-worker
    enter/exit pairs race (see the ``_BLAS_STATE`` comment above)."""
    prev = _get_blas_threads()
    _BLAS_STATE.n = None if n is None else max(1, int(n))
    try:
        yield
    finally:
        _BLAS_STATE.n = prev



# S5-8 (perf, no-loss): ``threadpool_limits(...)`` rebuilds a fresh
# ``ThreadpoolController`` -- and RE-ENUMERATES every loaded BLAS/OpenMP DLL
# (~9 ms on Windows) -- on EVERY call, so an N-wavelength sweep paid that DLL
# scan once per solve (a 20-wavelength RCWA sweep measured 283 -> 105 ms once
# cached).  The set of loaded BLAS libraries is fixed after import, so enumerate
# ONCE into a process-wide controller and reuse its ``.limit(...)`` (which
# applies the cap without re-scanning).  BIT-IDENTICAL: the same limiter
# save/restore runs, only the library discovery is amortised.
_BLAS_CONTROLLER = None
_BLAS_CONTROLLER_UNAVAILABLE = False
_BLAS_CONTROLLER_LOCK = threading.Lock()


def _get_blas_controller():
    """Return the process-wide cached ``ThreadpoolController`` (enumerated
    once), or ``None`` when ``threadpoolctl`` predates ``ThreadpoolController``
    (< 3.0) or is absent entirely.  The lazy first build is lock-guarded; the
    controller object is read-only shared state thereafter."""
    global _BLAS_CONTROLLER, _BLAS_CONTROLLER_UNAVAILABLE
    if _BLAS_CONTROLLER is not None:
        return _BLAS_CONTROLLER
    if _BLAS_CONTROLLER_UNAVAILABLE:
        return None
    with _BLAS_CONTROLLER_LOCK:
        if _BLAS_CONTROLLER is None and not _BLAS_CONTROLLER_UNAVAILABLE:
            try:
                from threadpoolctl import ThreadpoolController
            except ImportError:  # pragma: no cover - env-dependent optional dep
                _BLAS_CONTROLLER_UNAVAILABLE = True
                return None
            _BLAS_CONTROLLER = ThreadpoolController()
    return _BLAS_CONTROLLER


def _blas_limit():
    """Apply this thread's BLAS cap if one is set, else a zero-overhead no-op
    context (so the default path is untouched)."""
    n = _get_blas_threads()
    if n is None:
        return contextlib.nullcontext()
    controller = _get_blas_controller()
    if controller is not None:
        # Reuse the cached enumeration -- no per-solve DLL re-scan.
        return controller.limit(limits=n, user_api="blas")
    # threadpoolctl too old to expose ThreadpoolController: preserve the
    # legacy per-call path (re-enumerates, but keeps the cap) so the opt-in
    # behaviour is unchanged.  If threadpoolctl is missing ENTIRELY the cap is
    # inert -- set_blas_threads() has already warned once (audit M6).
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - env-dependent optional dep
        return contextlib.nullcontext()
    return threadpool_limits(limits=n, user_api="blas")



def _with_blas_limit(fn):
    """Decorator: run an RCWA entry point under the optional BLAS-thread cap."""
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _blas_limit():
            return fn(*args, **kwargs)
    return _wrapped
