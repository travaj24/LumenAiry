"""
Backend-aware FFT dispatch.

Provides a single public 2-D / 1-D FFT entry point that all
lumenairy modules can call regardless of the input array's backend.
The priority chain matches the long-standing behaviour of
:func:`lumenairy.propagation._fft2`:

    For CuPy arrays            -> ``cupy.fft`` (cuFFT)
    For JAX arrays             -> ``jax.numpy.fft`` (XLA)
    For NumPy arrays:
        1. **pyFFTW** (preferred) -- multi-threaded CPU FFT with
           per-shape plan caching.  Falls back automatically on
           allocation failure.
        2. **scipy.fft** -- multi-threaded ``workers=-1`` pocketfft.
        3. **numpy.fft** -- single-threaded fallback (always
           available).

JAX FFT dispatch is added on top of the existing chain so that
``fft2(jnp.ones(...))`` returns a JAX array suitable for
``jax.grad`` / ``jax.jit``.

This module is the public FFT entry point for new code.
``lumenairy.propagators.fft_infra`` (the post-v5.1-split home of the
plan cache + scipy thread pool + bad-shape blacklist) owns the
infrastructure; this module delegates into it for the NumPy / CuPy
paths so the priority chain and caches are shared, not duplicated.

v5.2 (ROADMAP "backend/fft.py -> propagators/propagation.py
inversion" cleanup): pre-v5.1, the FFT infra lived inside
``propagators/propagation.py`` and ``backend/fft.py`` had to import
through that monolith.  v5.1 lifted the infra to ``fft_infra.py``;
v5.2 routes ``backend/fft.py`` directly through ``fft_infra``
(``from ..propagators import fft_infra as _prop``) so the inversion
through the propagation shell is removed and ``__getattr__``
forwarding (PEP-562) no longer sits in the hot FFT path.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from .array import (
    _get_jnp,
    is_cupy_array,
    is_jax_array,
)

# Dead code removed in v5.29.1 (audit A-9..A-14): ``_jnp_or_none()`` had
# zero references repo-wide (grep-verified over every .py/.pyi/.md/.cfg/
# .toml in the tree -- the only hits were its own ``def`` line and the
# audit report naming it).  ``_jnp_required`` below is the live accessor;
# the ``JAX_AVAILABLE`` import went with it, having had no other use here.


def _jnp_required() -> Any:
    """Return jax.numpy or assert -- used after :func:`is_jax_array`
    has narrowed an argument to a JAX array (which guarantees JAX is
    installed).  v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure):
    centralises the None-narrow for the dispatch branches below so
    each call site doesn't need its own ``assert is not None``.
    """
    jnp = _get_jnp()
    assert jnp is not None, (
        "_jnp_required: jax.numpy unavailable yet is_jax_array() "
        "returned True -- inconsistent backend state."
    )
    return jnp


# ============================================================================
# 2-D FFT entry points
# ============================================================================

def fft2(x: Any) -> Any:
    """Backend-aware 2-D forward FFT on the last two axes.

    Dispatch order:

    * JAX array  -> ``jax.numpy.fft.fft2``
    * Otherwise (NumPy / CuPy)  -> :func:`lumenairy.propagation._fft2`
      which preserves the pyFFTW > scipy.fft > numpy.fft priority
      chain plus CuPy short-circuit.

    Returned-buffer ownership -- THE CALLER DOES NOT OWN THE RESULT
    ---------------------------------------------------------------
    v5.29.1 (audit P13-P16): this function forwards
    :func:`lumenairy.propagators.fft_infra._fft2` **unchanged**, and so
    inherits its double-buffer contract verbatim.  On the pyFFTW path
    (NumPy input, complex dtype, ``shape[0] >= FFTW_MIN_SIZE``, plan-cache
    double-buffer enabled -- the shipped default) the returned array is a
    view into one of two ping-pong workspace buffers owned by the plan
    cache, NOT a fresh allocation (``result.flags.owndata is False``).
    Therefore:

    * The result is guaranteed stable across exactly **ONE** subsequent
      same-shape FFT call.  The SECOND one silently overwrites it.
      Measured at N=512 complex128: unchanged after one further ``fft2``
      (max|delta| = 0.0), then ``max|delta| = 3.66e3`` on a field whose own
      peak magnitude is 2.45e3 after the second (``ifft2``: 1.40e-2
      against a 9.33e-3 peak).
    * Callers that keep the result alive across more than one further FFT
      -- storing it in a cache, a returned dataclass, a plane log, or a
      transfer function -- **must** ``.copy()`` it.  See
      ``propagators/rs.py`` (audit F-3) and ``propagators/fresnel.py``
      (audit P2) for the two in-library sites where exactly this bug was
      found and fixed with a ``.copy()``; P2's symptom was a
      previously-returned field silently becoming byte-identical to a
      later leg's result.
    * No copy is added here on purpose: in-library callers rely on the
      zero-copy path (it is the ~256 MB-1 GB per-call saving at 4k-8k
      grids that motivated the ping-pong), so the copy belongs at the
      sites that retain the array.  ``set_fft_double_buffer(False)`` makes
      every return a private copy globally if you want that trade.
    * The scipy / numpy fallback path, the CuPy path, and the JAX path all
      return fresh arrays, so a defensive ``.copy()`` is a harmless no-op
      there.
    """
    if is_jax_array(x):
        return _jnp_required().fft.fft2(x)
    # v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): fft_infra is
    # not on the typed-files whitelist so its public callables surface
    # as untyped under follow_imports=silent.
    from ..propagators import fft_infra as _prop
    return _prop._fft2(x)  # type: ignore[no-untyped-call]


def ifft2(x: Any) -> Any:
    """Backend-aware 2-D inverse FFT.  Same priority chain as
    :func:`fft2` -- and the SAME returned-buffer ownership contract: on
    the pyFFTW path the result is a view into the inverse ping-pong
    workspace, stable across exactly one subsequent same-shape call.
    Callers retaining it longer must ``.copy()``.  See :func:`fft2` for
    the measured numbers and the in-library precedents (audit P13-P16)."""
    if is_jax_array(x):
        return _jnp_required().fft.ifft2(x)
    from ..propagators import fft_infra as _prop
    return _prop._ifft2(x)  # type: ignore[no-untyped-call]


# ============================================================================
# 1-D FFT entry points
# ============================================================================

def fft(x: Any, axis: int = -1, n: Optional[int] = None) -> Any:
    """Backend-aware 1-D forward FFT."""
    if is_jax_array(x):
        return _jnp_required().fft.fft(x, n=n, axis=axis)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.fft(x, n=n, axis=axis)
    from ..propagators import fft_infra as _prop
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        import scipy.fft as _sp
        return _sp.fft(x, n=n, axis=axis, workers=_prop.SCIPY_FFT_WORKERS)
    return np.fft.fft(x, n=n, axis=axis)


def ifft(x: Any, axis: int = -1, n: Optional[int] = None) -> Any:
    """Backend-aware 1-D inverse FFT."""
    if is_jax_array(x):
        return _jnp_required().fft.ifft(x, n=n, axis=axis)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.ifft(x, n=n, axis=axis)
    from ..propagators import fft_infra as _prop
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        import scipy.fft as _sp
        return _sp.ifft(x, n=n, axis=axis, workers=_prop.SCIPY_FFT_WORKERS)
    return np.fft.ifft(x, n=n, axis=axis)


# ============================================================================
# FFT shifts and frequency grids
# ============================================================================

def fftshift(x: Any,
             axes: Optional[Union[int, Sequence[int]]] = None) -> Any:
    """Backend-dispatched ``fft.fftshift``."""
    if is_jax_array(x):
        return _jnp_required().fft.fftshift(x, axes=axes)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.fftshift(x, axes=axes)
    return np.fft.fftshift(x, axes=axes)


def ifftshift(x: Any,
              axes: Optional[Union[int, Sequence[int]]] = None) -> Any:
    """Backend-dispatched ``fft.ifftshift``."""
    if is_jax_array(x):
        return _jnp_required().fft.ifftshift(x, axes=axes)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.ifftshift(x, axes=axes)
    return np.fft.ifftshift(x, axes=axes)


def fftfreq(n: int, d: float = 1.0,
            xp: Optional[Any] = None) -> Any:
    """Backend-aware ``fft.fftfreq``.

    Pass ``xp`` to materialise the grid in a specific backend
    (CuPy GPU, JAX traced).  When ``xp`` is None, NumPy is used.
    """
    if xp is None:
        return np.fft.fftfreq(n, d=d)
    return xp.fft.fftfreq(n, d=d)


# ============================================================================
# Diagnostic
# ============================================================================

def fft_backend_for(x: Any) -> str:
    """Report which backend a call to :func:`fft2` would use for the
    given input.  Returns: ``'jax'`` / ``'cupy'`` / ``'pyfftw'`` /
    ``'scipy.fft'`` / ``'numpy.fft'``.

    S2-12 (audit AUDIT_V5_24_2): this diagnostic must mirror the ACTUAL
    :func:`fft_infra._fft2` dispatch.  ``_fft2`` gates the pyFFTW path on
    ``np.iscomplexobj(x)`` (the pyFFTW plan cache is complex-to-complex
    only -- a real-dtype array falls through to scipy/numpy), so a
    real-dtype input reported here as ``'pyfftw'`` was a lie.  Replicate
    the ``iscomplexobj`` gate so the report matches where the data really
    routes."""
    if is_jax_array(x):
        return 'jax'
    if is_cupy_array(x):
        return 'cupy'
    from ..propagators import fft_infra as _prop
    shape = tuple(x.shape) if hasattr(x, 'shape') else ()
    if (_prop.USE_PYFFTW and _prop.PYFFTW_AVAILABLE
            and np.iscomplexobj(x)
            and len(shape) >= 2
            and shape[0] >= _prop.FFTW_MIN_SIZE
            and shape not in _prop._PYFFTW_BAD_SHAPES):
        return 'pyfftw'
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        return 'scipy.fft'
    return 'numpy.fft'


__all__ = [
    'fft2', 'ifft2',
    'fft', 'ifft',
    'fftshift', 'ifftshift', 'fftfreq',
    'fft_backend_for',
]
