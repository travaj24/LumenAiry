"""
Backend-aware SciPy-compatible function dispatch.

A handful of lumenairy modules use ``scipy.special`` /
``scipy.linalg`` operations on array data.  When the input array is
JAX, those calls need to land in ``jax.scipy.*`` instead so the
operation stays in the JAX trace and is differentiable / JIT-able.

Dispatch by input array type:

    NumPy / CuPy arrays -> ``scipy.*`` (host) or ``cupyx.scipy.*``
    JAX arrays          -> ``jax.scipy.*``

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, cast

import scipy.linalg as _sp_linalg
import scipy.special as _sp_special

from .array import (
    JAX_AVAILABLE,
    is_cupy_array,
    is_jax_array,
)

_jax_special_cache = None
_jax_linalg_cache = None


def _get_jax_special() -> Optional[Any]:
    global _jax_special_cache
    if _jax_special_cache is None and JAX_AVAILABLE:
        import jax.scipy.special as _m
        _jax_special_cache = _m
    return _jax_special_cache


def _get_jax_linalg() -> Optional[Any]:
    global _jax_linalg_cache
    if _jax_linalg_cache is None and JAX_AVAILABLE:
        import jax.scipy.linalg as _m
        _jax_linalg_cache = _m
    return _jax_linalg_cache


# Property-style accessors so existing _jax_special / _jax_linalg
# references still work.  Resolved on first access; AttributeError
# behaviour mirrors the underlying module so getattr(default) calls
# still work as before.
class _LazyAttrProxy:
    def __init__(self, getter: Callable[[], Optional[Any]]) -> None:
        object.__setattr__(self, '_getter', getter)

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion on internal attrs.
        if name.startswith('_'):
            raise AttributeError(name)
        m = self._getter()
        if m is None:
            # JAX absent -- expose nothing, mirroring the original
            # ``_jax_special = None`` behaviour at the call site.
            raise AttributeError(name)
        return getattr(m, name)


_jax_special = _LazyAttrProxy(_get_jax_special)
_jax_linalg = _LazyAttrProxy(_get_jax_linalg)


def _dispatch_special(name: str, x: Any, *args: Any,
                      arg_pos: int = 0, **kwargs: Any) -> Any:
    """Generic dispatch for ``scipy.special`` functions on input ``x``.

    Parameters
    ----------
    name : str
        ``scipy.special`` function name.
    x : array-like
        The ARRAY argument -- the one whose backend selects the
        implementation.
    *args
        The remaining positional arguments, in their own order.
    arg_pos : int, keyword-only, default 0
        Where ``x`` sits in the target function's positional signature.
        ``0`` (the historical behaviour) calls ``fn(x, *args)``; for a
        two-argument special function whose ARRAY is the SECOND argument
        -- ``scipy.special.jv(v, z)``, ``kv(v, z)``, ``eval_legendre(n,
        x)`` -- pass ``arg_pos=1`` so the call becomes ``fn(args[0], x,
        *args[1:])``.

        Without this the helper hard-coded "the array is the first
        argument", which silently transposed
        :func:`jv`: ``jv(v=0, x=2.0)`` returned ``scipy.special.jv(2.0,
        0)`` = 0.000000000 instead of 0.223890779 (audit K2).
    """
    # The full positional argument tuple, with ``x`` restored to the
    # position the target function actually expects it in.
    call_args: Tuple[Any, ...] = (
        tuple(args[:arg_pos]) + (x,) + tuple(args[arg_pos:]))

    if is_jax_array(x):
        fn = getattr(_jax_special, name, None)
        if fn is None:
            raise NotImplementedError(
                f"jax.scipy.special.{name} is not available.")
        return fn(*call_args, **kwargs)
    if is_cupy_array(x):
        try:
            import cupyx.scipy.special as _cu_special
            fn = getattr(_cu_special, name, None)
            if fn is None:
                import cupy as cp
                host = tuple(cp.asnumpy(a) if is_cupy_array(a) else a
                             for a in call_args)
                return cp.asarray(
                    getattr(_sp_special, name)(*host, **kwargs))
            return fn(*call_args, **kwargs)
        except ImportError:
            import cupy as cp
            host = tuple(cp.asnumpy(a) if is_cupy_array(a) else a
                         for a in call_args)
            return cp.asarray(getattr(_sp_special, name)(*host, **kwargs))
    return getattr(_sp_special, name)(*call_args, **kwargs)


def jv(v: Any, x: Any) -> Any:
    """Bessel function of the first kind, order ``v``, evaluated at ``x``.

    Argument order matches :func:`scipy.special.jv` -- ORDER first,
    argument second.

    Examples
    --------
    >>> from lumenairy.backend.scipy import jv
    >>> float(jv(0, 2.0))                       # doctest: +ELLIPSIS
    0.2238907...
    >>> float(jv(2, 1.5))                       # doctest: +ELLIPSIS
    0.2320876...
    """
    if is_jax_array(x):
        if hasattr(_jax_special, 'bessel_jv'):
            return _jax_special.bessel_jv(v, x)
        raise NotImplementedError(
            "jv is not available in jax.scipy.special for arbitrary "
            "orders.  Convert to NumPy first.")
    # K2 (audit 2026-09-11): the array ``x`` is the SECOND positional
    # argument of ``scipy.special.jv(v, z)``.  The pre-fix call
    # ``_dispatch_special('jv', x, v)`` placed it first, i.e. it computed
    # ``scipy.special.jv(x, v)`` -- order and argument transposed --
    # returning a plausible wrong number with no diagnostic
    # (jv(0, 2.0) -> 0.000000000 instead of 0.223890779; jv(2, 1.5) ->
    # 0.491293779 instead of 0.232087672), while the JAX branch three
    # lines above was correct, so the two backends disagreed.
    return _dispatch_special('jv', x, v, arg_pos=1)


def erf(x: Any) -> Any:
    """Error function."""
    return _dispatch_special('erf', x)


def gammaln(x: Any) -> Any:
    """Log-gamma."""
    return _dispatch_special('gammaln', x)


def expi(x: Any) -> Any:
    """Exponential integral Ei."""
    return _dispatch_special('expi', x)


def solve(A: Any, b: Any) -> Any:
    """Solve ``A x = b`` on the appropriate backend."""
    if is_jax_array(A) or is_jax_array(b):
        return _jax_linalg.solve(A, b)
    if is_cupy_array(A) or is_cupy_array(b):
        import cupy as cp
        return cp.linalg.solve(A, b)
    return _sp_linalg.solve(A, b)


def lstsq(A: Any, b: Any, **kwargs: Any) -> Any:
    """Least-squares solve."""
    if is_jax_array(A) or is_jax_array(b):
        import jax.numpy as jnp
        return jnp.linalg.lstsq(A, b, **kwargs)
    if is_cupy_array(A) or is_cupy_array(b):
        import cupy as cp
        return cp.linalg.lstsq(A, b, **kwargs)
    return _sp_linalg.lstsq(A, b, **kwargs)


def eigh(A: Any) -> Tuple[Any, Any]:
    """Hermitian eigendecomposition."""
    # v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): the backend
    # ``eigh`` returns surface as ``Any`` (untyped backend modules under
    # follow_imports=silent), but each is documented to be a 2-tuple of
    # arrays.  Cast so the public return type is honoured.
    if is_jax_array(A):
        import jax.numpy as jnp
        return cast(Tuple[Any, Any], jnp.linalg.eigh(A))
    if is_cupy_array(A):
        import cupy as cp
        return cast(Tuple[Any, Any], cp.linalg.eigh(A))
    return cast(Tuple[Any, Any], _sp_linalg.eigh(A))


__all__ = ['jv', 'erf', 'gammaln', 'expi', 'solve', 'lstsq', 'eigh']
