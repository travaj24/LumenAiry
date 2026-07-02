"""Audit wave 6 (v5.17.0 deep audit) -- ui cluster.

[P3-62] WaveOpticsWorker declared ``finished = Signal(object)``,
shadowing QThread's built-in finished signal: the canonical
``worker.finished.connect(worker.deleteLater)`` idiom bound to the
custom signal (results payload silently discarded; never emitted at
all pre-wave-1 if run() raised).  Fixed by renaming the results signal
to ``finished_result``, matching all sibling dock workers.

[P3-63] WaveOpticsWorker.run read the live GUI-thread SystemModel
(wavelength_m, build_trace_surfaces(), source, to_prescription(),
lens_options, bfl_mm/efl_mm) at different times DURING the background
run, so a mid-run GUI edit produced mixed pre/post-edit state or a
racing-mutation crash.  Fixed by snapshotting all needed model state
in __init__ (GUI thread); run() reads only the snapshot.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

try:
    import matplotlib  # noqa: F401
    matplotlib.use('Agg')
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from lumenairy.ui.waveoptics_dock import WaveOpticsWorker
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'

pytestmark = pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)


class _FakeSurface:
    is_mirror = False
    is_coordbrk = False
    label = 'S'
    radius = float('inf')
    conic = 0.0
    semi_diameter = float('inf')
    thickness = 0.0
    glass_before = 'air'
    glass_after = 'air'


class _StubModel:
    wavelength_m = 550e-9
    epd_m = 5e-3
    elements = []
    source = None
    bfl_mm = float('nan')

    def build_trace_surfaces(self):
        return [_FakeSurface()]


def _cfg():
    return {'N': 32, 'dx_m': 1e-6, 'method': 'asm', 'backend': 'numpy'}


def _run_sync(worker):
    """Run synchronously with direct-connection signal collection."""
    QApplication.instance() or QApplication([])
    from lumenairy.propagators import fft_infra
    orig = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT)
    got = []
    worker.finished_result.connect(got.append)
    try:
        worker.run()
    finally:
        fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT = orig
    return got


# ── P3-62: signal rename ────────────────────────────────────────────

def test_builtin_qthread_finished_not_shadowed():
    """The class must NOT redeclare `finished`: QThread's built-in
    (always-fires, no-arg) signal must be the one visible on the
    class, so worker.finished.connect(worker.deleteLater) works."""
    assert 'finished' not in WaveOpticsWorker.__dict__
    assert WaveOpticsWorker.finished is QThread.finished


def test_results_delivered_on_finished_result():
    """The renamed results signal carries the results payload."""
    w = WaveOpticsWorker(_StubModel(), _cfg())
    got = _run_sync(w)
    assert len(got) == 1
    assert 'error' not in got[0]
    assert 'planes' in got[0]


# ── P3-63: GUI-thread snapshot ──────────────────────────────────────

def test_worker_does_not_read_live_model_after_init():
    """Mutating EVERY model attribute after construction (simulating a
    GUI edit racing the background run) must not affect the run: the
    worker reads only the __init__-time snapshot."""
    model = _StubModel()
    w = WaveOpticsWorker(model, _cfg())
    # Sabotage the live model post-construction.
    model.wavelength_m = 999e-9
    model.epd_m = None
    model.bfl_mm = object()
    model.elements = None
    model.build_trace_surfaces = lambda: (_ for _ in ()).throw(
        RuntimeError('mid-run mutation'))
    got = _run_sync(w)
    assert len(got) == 1
    assert 'error' not in got[0]
    assert got[0]['wavelength'] == 550e-9   # pre-edit value

def test_snapshot_taken_on_construction_thread():
    """build_trace_surfaces must be invoked during __init__ (GUI
    thread), not from run() on the worker thread."""
    calls = []

    class _Recorder(_StubModel):
        def build_trace_surfaces(self):
            calls.append(QThread.currentThread())
            return [_FakeSurface()]

    w = WaveOpticsWorker(_Recorder(), _cfg())
    assert len(calls) == 1          # snapshotted exactly once, in __init__
    _run_sync(w)
    assert len(calls) == 1          # run() never touched the model


def test_snapshot_failure_emits_error_payload_from_run():
    """A model property that raises during the snapshot must not escape
    __init__ on the GUI thread; run() reports it via the standard
    error payload so the dock re-enables the Run button."""

    class _BrokenModel(_StubModel):
        @property
        def wavelength_m(self):
            raise RuntimeError('broken prescription')

    w = WaveOpticsWorker(_BrokenModel(), _cfg())   # must not raise
    got = _run_sync(w)
    assert got == [{'error': 'RuntimeError: broken prescription'}]
