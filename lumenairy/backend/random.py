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
        self._key, sub = jax.random.split(self._key)
        if p is None:
            return jax.random.randint(sub, shape, 0, n)
        if not replace:
            raise NotImplementedError(
                "RandomState.choice with replace=False and weights is "
                "not implemented for the JAX backend.")
        return jax.random.choice(sub, n, shape=shape, p=p)


def _is_jax_prng_key(x) -> bool:
    if not JAX_AVAILABLE:
        return False
    import jax
    if hasattr(jax.random, 'KeyArray') and isinstance(x, jax.random.KeyArray):
        return True
    if is_jax_array(x):
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
