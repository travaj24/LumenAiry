"""
Shared optional-accelerator probes (CuPy / Numba).
==================================================

One place to ask "is CuPy installed?", "is this array a CuPy array?" and
"can I have Numba?".  Before this module the same three helpers were
hand-copied across eight modules -- ``_ensure_cupy_loaded`` x5,
``_is_cupy_array`` x5, ``_load_numba`` x5 (audit TESTS-ARCH P2-9,
2026-09-11) -- so the accelerator-absent path had five independent
implementations and no single place to test it.

Contract (CONVENTIONS.md section 10, reproduced exactly)
--------------------------------------------------------
* Availability is probed with :func:`importlib.util.find_spec`, which does
  NOT import the package, at this module's import time.  The two
  ``*_AVAILABLE`` constants are therefore free.
* The package itself is imported on FIRST USE and cached for the life of
  the process.  A second call is a dict/global read.
* Absence is reported as a value (``None`` / ``False``), never as an
  exception: every caller here has a pure-NumPy fallback and picks it by
  branching.  (Entry points with no fallback -- e.g.
  ``propagators/fga.py`` -- raise their own ``ImportError`` with the
  ``pip install lumenairy[...]`` hint; that is their contract, not this
  module's.)

Why this module is a LEAF
-------------------------
It imports nothing from lumenairy and nothing heavier than ``importlib``
so that the modules highest in the import order (``propagators/fft_infra``,
``sources/core``) can use it without dragging a dependency edge behind
them.  Do not add imports here.

Thread safety
-------------
The first-use import is not lock-guarded, matching every copy it replaces:
two threads racing the first :func:`ensure_cupy` both run ``import cupy``,
which CPython's own import lock serialises, and both then rebind the same
module object to the same global.  The observable result is identical.

Author: Andrew Traverso -- v5.45.2 (audit 2026-09-11 TESTS-ARCH P2-9).
"""

from __future__ import annotations

import importlib.util as _importlib_util
from typing import Any, Optional, Tuple

__all__ = [
    'CUPY_AVAILABLE',
    'NUMBA_AVAILABLE',
    'ensure_cupy',
    'is_cupy_array',
    'load_numba',
    'numba_handles',
]

# ---------------------------------------------------------------------------
# CuPy
# ---------------------------------------------------------------------------

#: True iff ``cupy`` is importable in this environment.  Probed with
#: ``find_spec`` at import time (no import, ~50 us); a CUDA box otherwise
#: pays ~150 ms of CuPy init for a NumPy-only run.
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None

#: The cupy module once :func:`ensure_cupy` has loaded it, else ``None``.
#: Read it through :func:`cupy_module`; the global is exposed only so a
#: consumer module can keep its own ``cp`` alias in sync.
_cp: Optional[Any] = None


def ensure_cupy() -> Optional[Any]:
    """Import CuPy on first use and return the module, or ``None``.

    Returns
    -------
    module or None
        The ``cupy`` module when it is installed (imported on the first
        call, cached afterwards), ``None`` when it is not.

    Notes
    -----
    Returning the MODULE rather than a bool is the one deliberate
    difference from the five copies this replaces (each of which returned
    ``cp is not None``): a caller that wants the bool writes
    ``ensure_cupy() is not None``, and a caller that wants the module no
    longer has to reach for a second global.  The consumer modules keep
    their historical ``_ensure_cupy_loaded() -> bool`` spelling as a thin
    wrapper, so no public behaviour changed.
    """
    global _cp
    if _cp is None and CUPY_AVAILABLE:
        import cupy as _c
        _cp = _c
    return _cp


def cupy_module() -> Optional[Any]:
    """Return the cached cupy module WITHOUT triggering the import.

    ``None`` both when CuPy is absent and when it is present but has not
    been loaded yet.  Use :func:`ensure_cupy` to load it.
    """
    return _cp


def is_cupy_array(x: Any) -> bool:
    """Return True iff ``x`` is a CuPy ``ndarray``.

    ``hasattr(x, 'device')`` is NOT a usable duck-type test: NumPy 2.x
    exposes ``ndarray.device`` as part of the Python Array API, so every
    NumPy array falsely tested as a CuPy array and got routed into the
    (unusable without CUDA) CuPy branch.  ``isinstance`` against the real
    type is the check.

    False -- without importing anything -- when CuPy is not installed, so
    this is safe to call on a hot path in a NumPy-only environment.
    """
    if not CUPY_AVAILABLE:
        return False
    cp = ensure_cupy()
    if cp is None:
        return False
    return isinstance(x, cp.ndarray)


# ---------------------------------------------------------------------------
# Numba
# ---------------------------------------------------------------------------

#: True iff ``numba`` is importable.  The eager ``import numba`` it
#: replaces cost ~1.8 s of ``import lumenairy`` cold start (audit P2-D).
NUMBA_AVAILABLE = _importlib_util.find_spec('numba') is not None

_numba: Optional[Any] = None
_njit: Optional[Any] = None
_prange: Optional[Any] = None


def load_numba() -> bool:
    """Import ``numba`` + ``njit`` + ``prange`` on first use; cache the handles.

    Returns
    -------
    bool
        True iff numba is importable.  False means the caller must take
        its pure-NumPy fallback -- every kernel in this library has one,
        so this is a value, not an error.

    Notes
    -----
    Byte-for-byte the semantics of the five copies it replaces, including
    the ``if _numba is not None: return True`` fast path (so a second call
    costs one global read and no ``find_spec``).
    """
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not NUMBA_AVAILABLE:
        return False
    import numba as _nb
    from numba import njit as _nj
    from numba import prange as _pr
    _numba, _njit, _prange = _nb, _nj, _pr
    return True


def numba_handles() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Return ``(numba, njit, prange)``, loading them on first use.

    ``(None, None, None)`` when numba is not installed.  Convenience for
    the kernel factories, which need ``njit`` and ``prange`` together:

    >>> nb, njit, prange = numba_handles()       # doctest: +SKIP
    >>> if njit is None:                          # doctest: +SKIP
    ...     return None                           # pure-NumPy fallback
    """
    if not load_numba():
        return None, None, None
    return _numba, _njit, _prange
