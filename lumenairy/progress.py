"""
Progress-reporting plumbing shared by long-running core operations.

Rationale
---------
Several functions in this package (``apply_real_lens_traced``,
``propagate_through_system``, ``through_focus_scan``, tolerance sweeps)
are slow enough on large grids that callers want to drive a progress
bar.  This module provides a tiny, dependency-free protocol that any
script or GUI can implement.

Usage from a script
-------------------
>>> from lumenairy import progress_callback, apply_real_lens_traced
>>> def my_cb(stage, frac, msg=''):
...     print(f'{stage}: {frac*100:.0f}% {msg}')
>>> E_out = apply_real_lens_traced(
...     E_in, rx, wavelength=1.3e-6, dx=2e-6,
...     progress=my_cb)

Usage from a Qt dock
--------------------
Connect a Qt ``Signal(str, float, str)`` to the callback and the
``progress_callback`` fires from a worker thread per stage.

Hook protocol
-------------
A callback is ``callable(stage: str, fraction: float, message: str='')``
where ``fraction`` is in [0.0, 1.0].  Implementations must be cheap
(no blocking I/O) and thread-safe.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Callable, Optional

#: Callback type alias.  ``stage`` is a free-form string ('surface_1/4',
#: 'raytrace', 'inversion', ...) that the caller can use for labeling
#: or bucketing.  ``fraction`` is overall progress in [0, 1].
ProgressCallback = Callable[[str, float, str], None]


def call_progress(cb: Optional[ProgressCallback],
                  stage: str, fraction: float,
                  message: str = '') -> None:
    """Invoke a progress callback if one is set, swallowing failures.

    Progress reporting must never break the underlying computation, so
    any exception in the callback is suppressed.  Callers pass their
    own callback through as an opt-in; ``None`` (the default) skips
    the whole mechanism with no overhead.
    """
    if cb is None:
        return
    try:
        cb(stage, float(fraction), message)
    except Exception:
        # Don't let a broken progress bar crash the simulation.
        pass


class ProgressScaler:
    """Nest sub-tasks within a parent progress budget.

    A scaler is both a two-argument inline sub-callback ``(frac, msg)``
    *and* a drop-in ``ProgressCallback`` so it can be passed straight
    into the ``progress=`` kwarg of a lower-level function:

    >>> parent_cb = lambda s, f, m: print(s, f, m)
    >>> child = ProgressScaler(parent_cb, 'real_lens', 0.2, 0.5)
    >>> child(0.5, '2/4 surfaces')            # inline form
    >>> child('inner', 0.5, '2/4 surfaces')   # protocol form (3-arg)

    Both forms rescale ``frac`` into the parent window ``[lo, hi]`` and
    forward under the scaler's own ``stage`` label (the outer task's
    name wins over whatever string the inner function emits, because
    the scaler represents the caller's roll-up view).
    """

    def __init__(self, parent: Optional[ProgressCallback], stage: str,
                 lo: float, hi: float) -> None:
        self.parent = parent
        self.stage = stage
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, *args) -> None:
        # Accept either (frac, msg='') -- inline sub-callback form,
        # or (stage, frac, msg='') -- ProgressCallback protocol form
        # used when the scaler is passed as ``progress=`` to a core
        # function.  The scaler's own ``self.stage`` is used in both
        # cases; the inner function's stage label is discarded.
        if len(args) == 3:
            _stage, frac, msg = args
        elif len(args) == 2:
            # Ambiguous: could be (frac, msg) or (stage, frac).  If
            # the first arg is a string, it's the protocol form with
            # no message; otherwise it's the inline form.
            if isinstance(args[0], str):
                _stage, frac = args
                msg = ''
            else:
                frac, msg = args
        elif len(args) == 1:
            frac, msg = args[0], ''
        else:
            raise TypeError(
                f'ProgressScaler expects 1, 2, or 3 positional args, '
                f'got {len(args)}')
        if self.parent is None:
            return
        overall = self.lo + (self.hi - self.lo) * max(0.0, min(1.0, float(frac)))
        call_progress(self.parent, self.stage, overall, msg)
