"""
Backend-aware random number sampling.

NumPy / CuPy use a stateful ``Generator``; JAX uses a functional
PRNG with explicit keys.  This module provides a thin shim so
HFPI / GBD / Monte-Carlo path sampling can be written once and run
on either idiom.

Usage::

    rs = RandomState(rng=42)              # numpy with int seed
    x = rs.uniform((1000, 2))

    rs = RandomState(rng=np.random.default_rng(42))
    x = rs.uniform((1000, 2))

    rs = RandomState(rng=jax.random.PRNGKey(42))
    x = rs.uniform((1000, 2))             # uses backend='jax'

The ``shape`` argument is given as a tuple in all backends.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from .array import (
    JAX_AVAILABLE,
    CUPY_AVAILABLE,
    is_jax_array,
)


class RandomState:
    """Backend-agnostic random state wrapper.

    Detects the underlying backend from the type of ``rng`` passed
    in: integer / None -> NumPy, ``np.random.Generator`` -> NumPy,
    ``cp.random.Generator`` -> CuPy, JAX ``PRNGKey`` -> JAX.

    For NumPy / CuPy the underlying generator is mutated by each
    draw (stateful idiom).  For JAX, ``rs.key`` advances after each
    draw via ``jax.random.split`` so chained calls don't repeat
    samples.
    """

    def __init__(self, rng: Optional[Union[int, object]] = None) -> None:
        if rng is None or isinstance(rng, (int, np.integer)):
            self._backend = 'numpy'
            self._rng = np.random.default_rng(None if rng is None else int(rng))
        elif isinstance(rng, np.random.Generator):
            self._backend = 'numpy'
            self._rng = rng
        elif _is_jax_prng_key(rng):
            self._backend = 'jax'
            self._key = rng
        elif _is_cupy_generator(rng):
            self._backend = 'cupy'
            self._rng = rng
        else:
            raise TypeError(
                f"RandomState: unrecognised rng of type {type(rng).__name__!r}."
                f"  Pass int seed, np.random.Generator, "
                f"cp.random.Generator, or jax.random.PRNGKey.")

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Sampling primitives.
    # ------------------------------------------------------------------

    def uniform(self, shape: Tuple[int, ...],
                low: float = 0.0, high: float = 1.0,
                dtype: Optional[Any] = None) -> Any:
        """Draw uniform samples on ``[low, high)``."""
        if self._backend == 'numpy':
            arr = self._rng.uniform(low, high, size=shape)
            return arr if dtype is None else arr.astype(dtype, copy=False)
        if self._backend == 'cupy':
            arr = self._rng.uniform(low, high, size=shape)
            return arr if dtype is None else arr.astype(dtype, copy=False)
        # JAX
        import jax
        self._key, sub = jax.random.split(self._key)
        if dtype is None:
            dtype = jax.numpy.float32
        return jax.random.uniform(sub, shape, dtype=dtype,
                                  minval=low, maxval=high)

    def normal(self, shape: Tuple[int, ...],
               mean: float = 0.0, std: float = 1.0,
               dtype: Optional[Any] = None) -> Any:
        """Draw standard-normal samples scaled by ``std`` and shifted
        by ``mean``."""
        if self._backend == 'numpy':
            arr = self._rng.normal(mean, std, size=shape)
            return arr if dtype is None else arr.astype(dtype, copy=False)
        if self._backend == 'cupy':
            arr = self._rng.normal(mean, std, size=shape)
            return arr if dtype is None else arr.astype(dtype, copy=False)
        import jax
        self._key, sub = jax.random.split(self._key)
        if dtype is None:
            dtype = jax.numpy.float32
        return mean + std * jax.random.normal(sub, shape, dtype=dtype)

    def integers(self, shape: Tuple[int, ...],
                 low: int = 0,
                 high: Optional[int] = None,
                 dtype: Optional[Any] = None) -> Any:
        """Draw uniform integers on ``[low, high)``."""
        if self._backend == 'numpy':
            arr = self._rng.integers(low, high, size=shape,
                                     dtype=dtype or np.int64)
            return arr
        if self._backend == 'cupy':
            arr = self._rng.integers(low, high, size=shape,
                                     dtype=dtype or np.int64)
            return arr
        import jax
        self._key, sub = jax.random.split(self._key)
        if dtype is None:
            dtype = jax.numpy.int32
        return jax.random.randint(sub, shape, low, high, dtype=dtype)

    def choice(self, n: int, shape: Tuple[int, ...],
               p: Optional[Any] = None,
               replace: bool = True) -> Any:
        """Draw integer indices from ``[0, n)`` with optional
        weights ``p``."""
        if self._backend == 'numpy':
            return self._rng.choice(n, size=shape, replace=replace, p=p)
        if self._backend == 'cupy':
            return self._rng.choice(n, size=shape, replace=replace, p=p)
        import jax
        import jax.numpy as jnp
        self._key, sub = jax.random.split(self._key)
        # v4.13.1 (P1-F): honour replace=False on the JAX backend.
        # Previously the ``p is None`` branch dispatched to
        # ``jax.random.randint``, which is always with-replacement; a
        # caller passing ``replace=False`` silently got with-replacement
        # samples.  Route through ``jax.random.choice(..., replace=False)``
        # whenever ``replace=False`` so the without-replacement contract
        # is honoured for both weighted and unweighted draws.
        if not replace:
            # v4.13.2 (P1-NEW-K): wrap the replace=False dispatch in
            # a try/except so pre-0.4.x JAX builds that lacked
            # ``replace=False`` on ``jax.random.choice`` raise a
            # clear migration error instead of a bare TypeError.
            # The v4.13.1 CHANGELOG promised this safety net; v4.13.2
            # ships it.
            try:
                return jax.random.choice(sub, n, shape=shape,
                                         replace=False, p=p)
            except TypeError as e:
                raise RuntimeError(
                    "RandomState.choice(replace=False) on JAX "
                    "requires JAX >= 0.4.0 (got TypeError).  Update "
                    "JAX or switch to backend='numpy'.") from e
        # v4.13.2 (P1-NEW-I): pin the int dtype to int64 so the JAX
        # branch matches the NumPy branch.  ``jax.random.randint``
        # defaults to int32 (and on x32-only JAX builds int64 is
        # silently demoted, but on x64-enabled builds the difference
        # broke cross-backend pipelines that downcast to int32).
        if p is None:
            return jax.random.randint(sub, shape, 0, n, dtype=jnp.int64)
        return jax.random.choice(sub, n, shape=shape, p=p)


def _is_jax_prng_key(x) -> bool:
    """Detect a JAX PRNG key in either the legacy uint32 form or the
    JAX 0.4.20+ opaque-dtype form (``jax.random.key(...)``).

    v4.13.1 (P3 #20): extended to recognise opaque keys.  The legacy
    typed form returns shape ``(..., 2)`` with dtype uint32; the opaque
    form returns scalar shape with a custom dtype whose ``name``
    starts with ``'key<'`` (e.g. ``'key<fry>'``).  When available,
    ``jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`` is the canonical
    check and is preferred.
    """
    if not JAX_AVAILABLE:
        return False
    import jax
    if hasattr(jax.random, 'KeyArray') and isinstance(x, jax.random.KeyArray):
        return True
    if is_jax_array(x):
        # First, try the canonical opaque-key check via jax.dtypes.
        # This is the preferred path on JAX 0.4.20+.
        try:
            dtypes_mod = getattr(jax, 'dtypes', None)
            if (dtypes_mod is not None
                    and hasattr(dtypes_mod, 'prng_key')
                    and hasattr(dtypes_mod, 'issubdtype')):
                if dtypes_mod.issubdtype(x.dtype, dtypes_mod.prng_key):
                    return True
        except (AttributeError, TypeError):
            pass
        # Fallback string check for opaque-key dtypes when the canonical
        # API is unavailable -- the dtype.name is ``key<impl>`` (e.g.
        # ``key<fry>``).
        try:
            dt_name = getattr(x.dtype, 'name', '')
            if isinstance(dt_name, str) and dt_name.startswith('key<'):
                return True
        except (AttributeError, TypeError):
            pass
        # Legacy typed (uint32, shape-trailing-2) form.
        try:
            return x.dtype == jax.numpy.uint32 and x.shape[-1] == 2
        except (AttributeError, IndexError, TypeError):
            return False
    return False


def _is_cupy_generator(x) -> bool:
    if not CUPY_AVAILABLE:
        return False
    try:
        import cupy as cp
        return isinstance(x, cp.random.Generator)
    except (ImportError, AttributeError):
        # cupy import-time guard for stale CUPY_AVAILABLE flag.
        return False


__all__ = ['RandomState']
