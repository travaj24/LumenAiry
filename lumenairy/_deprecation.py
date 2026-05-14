"""Shared deprecation-warning utilities for 4.7+ API-consistency work.

The 4.7 polish pass adopted the standard Python deprecation pattern --
warn for one minor release, change defaults next, remove later --
across the API-consistency items called out in
``lumenairy_4_6_polish_pass.md``.  This module centralises the
mechanics so each warning has a uniform message format and the
warnings can all be silenced with a single ``warnings.simplefilter``
incantation in user code.

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
import warnings
from typing import Any, Callable, Optional


__all__ = [
    'warn_deprecated_kwarg',
    'warn_deprecated_alias',
    'deprecated_alias',
    'warn_renamed_function',
    'warn_deprecated_default',
]


class _Sentinel:
    """Distinct singleton used as a default-argument marker.

    ``arg=_NO_DEFAULT`` lets callers distinguish "didn't pass a value"
    from "passed None" -- needed to warn-on-default-use without
    breaking explicit ``None`` callers.
    """
    __slots__ = ('_name',)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f'<{self._name}>'

    def __bool__(self) -> bool:  # noqa: D401 — sentinel is always falsy
        return False


_NO_DEFAULT = _Sentinel('NO_DEFAULT')


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
    stacklevel : int
        Passed through to ``warnings.warn``.
    """
    removal = (f', will be removed in v{version_removed}'
               if version_removed else '')
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
    removal = (f', will be removed in v{version_removed}'
               if version_removed else '')
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
    def _shim(*args, **kwargs):
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
                    version_removed='5.0',
                )
                wavelength = 550e-9
            ...
    """
    removal = (f', will be required in v{version_removed}'
               if version_removed else '')
    _emit(
        f"{function}: relying on the default value of '{arg_name}' "
        f"({default_value!r}) is deprecated since v{version_added}"
        f"{removal}; pass it explicitly.",
        stacklevel=stacklevel,
    )
