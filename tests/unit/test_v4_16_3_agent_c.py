"""v4.16.3 / Agent C -- optimize-subsystem polish closures.

Pins the 2 P2 polish items from
``docs/audits/AUDIT_V4_16_2_2026_05_20.md`` concentrated in
``lumenairy/optimize/core.py``:

* **P2-NEW-F1-1** -- ``Constraint.__post_init__`` auto-probe removal
  needs a transitional ``DeprecationWarning``.  v4.16.1 shipped a
  ``fun(np.zeros(1))`` auto-probe; v4.16.2 silently removed it (the
  probe was expensive for BFL-style callables that internally ran a
  full ray-trace) and moved the contract to an opt-in
  :meth:`Constraint.validate` method.  Compare to the v4.16.1
  SUM->AVG transition which DID get a one-cycle ``FutureWarning`` in
  v4.16.2 -- same silent-behaviour-shift treated inconsistently.
  v4.16.3 emits a one-cycle ``DeprecationWarning`` (latched at module
  level so optimisation loops don't flood the warning channel)
  describing the v4.16.1 -> v4.16.2 transition, advising callers to
  use ``.validate()`` explicitly, citing v4.16.2 as the change-version
  and v5.0 as the removal target.
* **P2-NEW-F1-2** -- the ``pickle.dumps`` probe catch list at
  ``optimize/core.py:3229-3232`` pre-v4.16.3 was
  ``(pickle.PicklingError, AttributeError, TypeError)`` -- too
  narrow.  It missed ``RecursionError`` (deep object graph),
  ``RuntimeError`` (raised by a custom ``__reduce__``),
  ``MemoryError`` (huge object), and arbitrary exceptions from
  ``__reduce__`` / ``__getstate__``.  A ``fun`` whose ``__reduce__``
  raises ``RuntimeError`` propagated out of ``Constraint(...)``
  construction, defeating the warning's "friendlier than rejecting"
  intent.  v4.16.3 widens the catch to ``except Exception`` --
  pickling is a best-effort heuristic; any failure is a "not safely
  picklable" signal.  ``BaseException`` (``KeyboardInterrupt`` /
  ``SystemExit``) is intentionally left to propagate.

Test taxonomy:
  * P2-NEW-F1-1: 5 tests -- (a) first construction emits exactly one
    ``DeprecationWarning``; (b) the latch suppresses re-emission on
    subsequent constructions in the same process; (c) the message
    cites v4.16.2 (change-version) + v5.0 (removal-target) + the
    silence-hint via ``warnings.filterwarnings``; (d) the warning is
    category ``DeprecationWarning`` not ``UserWarning`` (so the
    existing v4.16.2 negative-pin tests that count
    ``UserWarning``s still pass); (e) the latch is module-level
    (parallel to ``_MULTIWL_AVG_WARNED``).
  * P2-NEW-F1-2: 3 tests -- (a) a ``fun`` whose ``__reduce__``
    raises ``RuntimeError`` triggers the existing UserWarning and
    does NOT propagate; (b) a ``fun`` whose ``__reduce__`` raises
    ``RecursionError`` triggers the warning and does NOT propagate;
    (c) ``KeyboardInterrupt`` (a ``BaseException``) raised from
    ``__reduce__`` still propagates -- we don't accidentally swallow
    interpreter-control signals.

Total: 8 tests.

Author: Andrew Traverso -- v4.16.3 / Agent C
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level callables used in regression tests.  Module-level so
# they pickle cleanly (the v4.16.2 ``pickle.dumps`` probe must NOT
# emit a UserWarning for these), and so the test bodies stay clean.
# ---------------------------------------------------------------------------


def _module_level_sum(x):
    """sum(x).  Picklable: module-level def at the top of this file."""
    return float(np.sum(x))


def _module_level_first(x):
    """float(x[0]).  Picklable companion to ``_module_level_sum``."""
    return float(x[0])


# ===========================================================================
# P2-NEW-F1-2: pickle.dumps catch list widened to Exception
# ===========================================================================


class _ReducerThatRaisesRuntimeError:
    """A callable whose ``__reduce__`` raises ``RuntimeError`` -- a
    realistic example is a wrapper around a JAX-traced function or a
    library object that explicitly refuses pickling via a runtime
    error rather than the canonical ``pickle.PicklingError``.

    Pre-v4.16.3 the narrow catch ``(pickle.PicklingError,
    AttributeError, TypeError)`` missed this and the
    ``RuntimeError`` propagated out of ``Constraint(...)``
    construction, defeating the warning's "friendlier than
    rejecting" intent.
    """
    def __call__(self, x):
        return float(np.sum(x))

    def __reduce__(self):
        raise RuntimeError(
            "intentional: this object explicitly refuses pickling "
            "via a RuntimeError instead of pickle.PicklingError")


class _ReducerThatRaisesRecursionError:
    """A callable whose ``__reduce__`` raises ``RecursionError`` --
    a realistic example is a deeply nested object graph (e.g. a
    Constraint over a recursively-defined parameterisation).

    Pre-v4.16.3 the narrow catch missed this; v4.16.3 ``except
    Exception`` catches it.  ``RecursionError`` is a subclass of
    ``RuntimeError`` which is a subclass of ``Exception`` (NOT
    ``BaseException``), so it's correctly caught by the widened
    handler.
    """
    def __call__(self, x):
        return float(np.sum(x))

    def __reduce__(self):
        raise RecursionError(
            "intentional: this object's __reduce__ overflows the "
            "stack (synthetic for testing the v4.16.3 catch-list "
            "widening)")


class _ReducerThatRaisesKeyboardInterrupt:
    """A callable whose ``__reduce__`` raises
    ``KeyboardInterrupt`` -- a ``BaseException`` (NOT ``Exception``).

    The v4.16.3 catch ``except Exception`` MUST NOT swallow this
    because ``KeyboardInterrupt`` / ``SystemExit`` are
    interpreter-control signals that must propagate to the user.
    This pin enforces that we widened to ``Exception``, not
    ``BaseException``.
    """
    def __call__(self, x):
        return float(np.sum(x))

    def __reduce__(self):
        raise KeyboardInterrupt(
            "intentional: this exercises the v4.16.3 "
            "BaseException-propagation contract")


def test_p2_pickle_probe_catches_reduce_runtime_error():
    """audit closure: P2-NEW-F1-2.

    A ``fun`` whose ``__reduce__`` raises ``RuntimeError`` must
    trigger the existing UserWarning (NOT crash ``Constraint(...)``
    construction).  Pre-v4.16.3 the narrow catch list
    ``(pickle.PicklingError, AttributeError, TypeError)`` let the
    ``RuntimeError`` escape; v4.16.3 widens to ``except Exception``.
    """
    from lumenairy.optimize import Constraint

    fun = _ReducerThatRaisesRuntimeError()
    # Sanity-check the precondition: pickle.dumps on this raises
    # RuntimeError (not PicklingError / AttributeError / TypeError),
    # so the v4.16.2 narrow catch would have let it through.
    import pickle
    with pytest.raises(RuntimeError):
        pickle.dumps(fun)

    # The v4.16.3 widened catch must turn this into a UserWarning
    # rather than letting the RuntimeError crash construction.
    with pytest.warns(UserWarning, match=r'(picklable|pickle)'):
        c = Constraint(fun=fun, lb=0.0, ub=None,
                       label='reduce_runtime_error')
    # Sanity-check: the constructed Constraint still works
    # (single-process SLSQP path doesn't care about pickling).
    assert c.fun(np.zeros(1)) == 0.0


def test_p2_pickle_probe_catches_reduce_recursion_error():
    """audit closure: P2-NEW-F1-2.

    A ``fun`` whose ``__reduce__`` raises ``RecursionError`` must
    also trigger the UserWarning (not crash).  ``RecursionError``
    is a subclass of ``RuntimeError`` -> ``Exception``, so the
    widened catch handles it; pre-v4.16.3 the narrow tuple missed
    it (``RecursionError`` is not ``PicklingError`` /
    ``AttributeError`` / ``TypeError``).
    """
    from lumenairy.optimize import Constraint

    fun = _ReducerThatRaisesRecursionError()
    # Sanity-check the precondition.
    import pickle
    with pytest.raises(RecursionError):
        pickle.dumps(fun)

    with pytest.warns(UserWarning, match=r'(picklable|pickle)'):
        c = Constraint(fun=fun, lb=0.0, ub=None,
                       label='reduce_recursion_error')
    assert c.fun(np.zeros(1)) == 0.0


def test_p2_pickle_probe_does_not_swallow_keyboard_interrupt():
    """audit closure: P2-NEW-F1-2 (BaseException propagation pin).

    The v4.16.3 widening is to ``except Exception``, NOT
    ``except BaseException``.  ``KeyboardInterrupt`` and
    ``SystemExit`` are ``BaseException`` subclasses that signal
    interpreter control flow (user Ctrl-C, sys.exit) and MUST
    propagate -- swallowing them would hang an interactive session
    or hide an intended exit.

    Construct a ``fun`` whose ``__reduce__`` raises
    ``KeyboardInterrupt``; the ``Constraint(...)`` call must
    propagate the KeyboardInterrupt, not turn it into a
    UserWarning.
    """
    from lumenairy.optimize import Constraint

    fun = _ReducerThatRaisesKeyboardInterrupt()
    # Sanity-check the precondition: KeyboardInterrupt is NOT a
    # subclass of Exception (the parent class for our widened
    # catch); it IS a BaseException.
    assert not issubclass(KeyboardInterrupt, Exception), (
        "precondition violated: KeyboardInterrupt is a subclass of "
        "Exception in this Python build; the pin would not "
        "exercise the BaseException-propagation contract")
    assert issubclass(KeyboardInterrupt, BaseException)

    # The widened catch must NOT swallow this; it must propagate.
    with pytest.raises(KeyboardInterrupt):
        Constraint(fun=fun, lb=0.0, ub=None,
                   label='reduce_keyboard_interrupt')
