"""v5.4.2: pin SystemModel.run_trace() empty-prescription cursor-hang fix.

User-reported GUI hang: clicking Retrace with no prescription loaded
left the wait cursor + "Tracing..." status set forever.  Root cause:
``run_trace`` emitted ``trace_started`` BEFORE the
``if not surfaces: return None`` early-exit, so the cursor was set
but ``trace_ready`` was never emitted to restore it.

v5.4.2 (this fix): build the surface list first, emit
``trace_ready.emit(None)`` on the empty-prescription path, and only
emit ``trace_started`` after we know there's actual work to do.

Tests verify the signal-ordering contract on the empty path without
needing a Qt event loop (we mock the signals via lightweight
collector objects).
"""
from __future__ import annotations

import sys
import types

import pytest


def _build_signal_collector():
    """Return an object exposing a Signal-compatible ``emit`` whose
    calls are recorded on ``.calls``.
    """
    collector = types.SimpleNamespace(calls=[])
    collector.emit = lambda *args: collector.calls.append(args)
    collector.connect = lambda *_a, **_kw: None
    return collector


@pytest.fixture
def stub_model():
    """Construct a minimal SystemModel-like object that exercises the
    empty-prescription path of ``run_trace`` without needing a full
    Qt + PySide6 environment.

    We import the real SystemModel.run_trace method but bind it to a
    stub instance carrying only the attributes that path touches.
    """
    try:
        # Defer-import so this test does not need Qt unless run.
        # Importing PySide6 indirectly via lumenairy.ui.model is
        # required because Signal types are defined at class level.
        from lumenairy.ui.model import SystemModel
    except ImportError:
        pytest.skip('PySide6 / lumenairy.ui.model unavailable; '
                    'cannot pin empty-prescription run_trace.')

    # Subclass SystemModel for the test so we get a real instance with
    # all signals + state initialised, then replace ``elements`` with
    # an empty list to simulate the no-prescription state.
    sm = SystemModel()
    sm.elements = []           # force empty-prescription
    return sm


def test_run_trace_empty_prescription_emits_trace_ready_none(stub_model):
    """v5.4.2: empty prescription must emit trace_ready(None) on the
    early-exit path (replaces the cursor-hang bug)."""
    # Replace the model's signals with collectors so we can observe
    # emit() calls.
    started = _build_signal_collector()
    ready = _build_signal_collector()
    stub_model.trace_started = started
    stub_model.trace_ready = ready

    result = stub_model.run_trace()

    assert result is None, 'empty-prescription must return None'
    assert started.calls == [], (
        'trace_started must NOT fire on empty-prescription path '
        '(would set wait cursor without restoring it)')
    assert ready.calls == [(None,)], (
        f'trace_ready must fire exactly once with None on empty '
        f'prescription; got {ready.calls!r}')


def test_run_trace_empty_prescription_does_not_set_wait_cursor(stub_model):
    """v5.4.2: verify the empty-prescription path doesn't emit the
    trace_started signal at all.  Paired with the v3.7.9 _on_trace_started
    handler which sets the wait cursor unconditionally on emit; if
    trace_started doesn't fire, no cursor is set, no hang appearance."""
    started = _build_signal_collector()
    ready = _build_signal_collector()
    stub_model.trace_started = started
    stub_model.trace_ready = ready

    stub_model.run_trace()
    stub_model.run_trace()      # call twice; verify still no started.emit
    stub_model.run_trace()

    assert started.calls == [], (
        'trace_started must not fire on any empty-prescription run_trace '
        f'invocation; got {len(started.calls)} emit(s)')
    assert ready.calls == [(None,), (None,), (None,)], (
        f'trace_ready must fire once per invocation with None; '
        f'got {ready.calls!r}')
