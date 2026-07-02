"""v5.17.0 deep-audit wave 5 -- UI robustness cluster.

Findings under test (docs/audits/AUDIT_V5_17_0_2026_07_01_DEEP.md):

P2-37  coherence_dock._dispatch_worker rebound the shared
       _analysis_worker while a previous QThread was still running
       (cross-tab launch) -> the orphaned thread hard-aborts the app
       ('QThread: Destroyed while thread is still running') when its
       run() returns.  Fix: sibling-dock isRunning() guard.
P2-38  MainWindow.closeEvent only saved workspace preferences; no
       dock worker was interrupted/joined on app quit (dock-level
       closeEvent does NOT fire for docked widgets at quit, so even
       coronagraph_dock's v5.4.2 C1 guard never ran then).  Fix:
       central _shutdown_dock_workers sweep called from closeEvent.
P2-39  spot_field_dock recomputed the full multi-field spot trace +
       matplotlib rebuild on every system_changed even while hidden/
       tabified.  Fix: isVisible() gate + stale-flag +
       recompute-on-showEvent.
P2-40  waveoptics_dock._stop / through_focus_dock._stop_scan used
       QThread.terminate() (thread may hold the pyFFTW planner lock /
       be mid-HDF5 write) plus a manual _on_finished call (double-
       completion race + reference dropped while the thread was still
       dying).  Fix: cooperative requestInterruption() polled at stage
       boundaries; the worker's own finished emission is the only UI
       re-enable path.
F821 (wave-1 pre-existing)  _mhs_run_pipeline carried a stray paste-
       duplicate of _on_save_toggle's body referencing the undefined
       name ``checked`` -> NameError on every successful MHS pipeline
       run.  Fix: stray block removed (the sync lives, correctly, in
       _on_save_toggle).

All tests are headless (QT_QPA_PLATFORM=offscreen, matplotlib Agg) and
skip cleanly when the [gui] extra is not installed.
"""
import ast
import inspect
import os
import textwrap
import threading
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use('Agg')
    from PySide6.QtCore import QObject, Qt, QThread, Signal
    from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

    import lumenairy.ui.coherence_dock as coherence_dock
    import lumenairy.ui.spot_field_dock as spot_field_dock
    import lumenairy.ui.through_focus_dock as through_focus_dock
    import lumenairy.ui.waveoptics_dock as waveoptics_dock
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:  # pragma: no cover - CI without [gui] extra
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'
    # Headless CI (no [gui] extra) has no PySide6, so the names imported
    # above are undefined -- yet the helper classes further down subclass
    # QThread / QObject / a dock worker and declare ``Signal(...)`` class
    # attributes, all of which execute at COLLECTION time (before the
    # ``pytestmark`` skip can take effect).  Provide inert fallbacks so the
    # module still imports; every test is skipped via ``pytestmark`` so none
    # of these placeholders is ever instantiated or called.
    QObject = QThread = QApplication = QDockWidget = QMainWindow = \
        QWidget = object
    Qt = None

    def Signal(*_a, **_k):  # noqa: N802 - mimics PySide6.QtCore.Signal
        return None

    class _MissingDockModule:
        """A stand-in dock module whose every attribute (e.g. the
        ``ThroughFocusWorker`` used as a base class) resolves to ``object``
        so the helper class bodies below parse on a headless runner."""

        def __getattr__(self, _name):
            return object

    coherence_dock = spot_field_dock = through_focus_dock = \
        waveoptics_dock = _MissingDockModule()

# main_window pulls the whole dock fleet (incl. pyvista via layout_3d),
# so probe it separately: its absence must only skip the P2-38 tests.
try:
    import lumenairy.ui.main_window as main_window
    _MW_OK = _GUI_OK
    _MW_SKIP = _SKIP_REASON
except Exception as _e:  # pragma: no cover
    _MW_OK = False
    _MW_SKIP = f'main_window unavailable: {_e}'

pytestmark = pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)


def _app():
    return QApplication.instance() or QApplication([])


def _join(*threads, timeout_ms=5000):
    """Best-effort cleanup of helper QThreads (never leave one running)."""
    for th in threads:
        if isinstance(th, QThread) and th.isRunning():
            th.requestInterruption()
            th.wait(timeout_ms)


def _method_calls(func, attr_name):
    """AST-level: does *func*'s body contain a ``<x>.<attr_name>(...)``
    call?  (String matching would trip on the explanatory comments.)"""
    src = textwrap.dedent(inspect.getsource(func))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr_name
        for node in ast.walk(ast.parse(src)))


# ═══════════════════════════════════════════════════════════════════
# F821: _mhs_run_pipeline stray unbound `checked`
# ═══════════════════════════════════════════════════════════════════

def test_mhs_run_pipeline_has_no_unbound_checked():
    """Pre-fix: `_mhs_run_pipeline` loaded the name ``checked`` (a
    paste-duplicate of _on_save_toggle's body) without ever binding it
    -> NameError on every successful MHS pipeline run."""
    src = textwrap.dedent(
        inspect.getsource(waveoptics_dock.WaveOpticsDock._mhs_run_pipeline))
    fn = ast.parse(src).body[0]
    loads = {n.id for n in ast.walk(fn)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stores = {n.id for n in ast.walk(fn)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    params = {a.arg for a in fn.args.args}
    assert 'checked' not in (loads - stores - params)


def test_on_save_toggle_still_owns_the_sync():
    """The save-toggle <-> checkbox sync must remain in the handler
    that legitimately receives `checked` (the fix deleted the stray
    copy, not the real one)."""
    src = textwrap.dedent(
        inspect.getsource(waveoptics_dock.WaveOpticsDock._on_save_toggle))
    fn = ast.parse(src).body[0]
    assert 'checked' in {a.arg for a in fn.args.args}
    assert _method_calls(
        waveoptics_dock.WaveOpticsDock._on_save_toggle, 'setChecked')


# ═══════════════════════════════════════════════════════════════════
# P2-37: coherence_dock shared-worker rebind guard
# ═══════════════════════════════════════════════════════════════════

class _BlockingAnalysisWorker(QThread):
    """Signature-compatible stand-in for _CoherenceAnalysisWorker that
    runs until interrupted, so a 'previous run still active' state can
    be held deterministically."""
    finished_result = Signal(object)

    def __init__(self, sm, mode, params):
        super().__init__()
        self.mode = mode

    def run(self):
        while not self.isInterruptionRequested():
            self.msleep(5)


class _StubCoherenceModel:
    wavelength_m = 550e-9


@pytest.fixture
def coherence_dock_patched(monkeypatch):
    _app()
    monkeypatch.setattr(coherence_dock, '_CoherenceAnalysisWorker',
                        _BlockingAnalysisWorker)
    dock = coherence_dock.CoherenceDock(_StubCoherenceModel())
    try:
        yield dock
    finally:
        _join(dock._analysis_worker, dock._worker)


def test_coherence_dispatch_refuses_relaunch_while_running(
        coherence_dock_patched):
    """Launching Tab 4 while Tab 2's shared worker is still running
    must NOT rebind _analysis_worker (pre-fix it did, dropping the only
    reference to a live QThread -> app abort when its run() returned).
    """
    dock = coherence_dock_patched
    dock._run_koehler()
    w1 = dock._analysis_worker
    assert isinstance(w1, QThread)
    deadline = time.time() + 5
    while not w1.isRunning() and time.time() < deadline:
        time.sleep(0.01)
    assert w1.isRunning()

    dock._run_mcf()                      # cross-tab launch mid-run
    assert dock._analysis_worker is w1   # NOT rebound
    assert 'still' in dock.summary_mcf.toPlainText().lower()
    # The refused launch must not have disabled Tab 4's Run button
    # (the guard returns before touching the UI).
    assert dock.btn_mcf.isEnabled()


def test_coherence_dispatch_allows_new_run_after_finish(
        coherence_dock_patched):
    """The guard must clear once the previous worker finishes (i.e. it
    is an isRunning() check, not a one-shot latch)."""
    dock = coherence_dock_patched
    dock._run_koehler()
    w1 = dock._analysis_worker
    _join(w1)
    assert not w1.isRunning()
    dock._run_mcf()
    assert dock._analysis_worker is not w1
    assert dock._analysis_worker.isRunning()


# ═══════════════════════════════════════════════════════════════════
# P2-38: app-close worker shutdown sweep
# ═══════════════════════════════════════════════════════════════════

class _LoopWorker(QThread):
    def run(self):
        while not self.isInterruptionRequested():
            self.msleep(5)


@pytest.mark.skipif(not _MW_OK, reason=_MW_SKIP)
def test_shutdown_dock_workers_interrupts_and_joins():
    """The central sweep must find running QThreads on both `_worker`
    and `_analysis_worker` dock attributes, request interruption, and
    join them (bounded)."""
    _app()
    win = QMainWindow()
    w1, w2 = _LoopWorker(), _LoopWorker()
    for attr, th in (('_worker', w1), ('_analysis_worker', w2)):
        holder = QWidget()
        setattr(holder, attr, th)
        d = QDockWidget('t', win)
        d.setWidget(holder)
        win.addDockWidget(Qt.LeftDockWidgetArea, d)
        th.start()
    try:
        deadline = time.time() + 5
        while not (w1.isRunning() and w2.isRunning()) \
                and time.time() < deadline:
            time.sleep(0.01)
        found = main_window._shutdown_dock_workers(
            win, per_worker_timeout_ms=5000)
        assert set(found) == {w1, w2}
        assert not w1.isRunning()
        assert not w2.isRunning()
    finally:
        _join(w1, w2)


@pytest.mark.skipif(not _MW_OK, reason=_MW_SKIP)
def test_shutdown_dock_workers_ignores_idle_and_missing():
    """Docks with no worker attribute, a None worker, or a finished
    worker must be skipped (nothing to join, no AttributeError)."""
    _app()
    win = QMainWindow()
    idle = QWidget()
    idle._worker = None
    bare = QWidget()
    for holder in (idle, bare):
        d = QDockWidget('t', win)
        d.setWidget(holder)
        win.addDockWidget(Qt.LeftDockWidgetArea, d)
    assert main_window._shutdown_dock_workers(win) == []


@pytest.mark.skipif(not _MW_OK, reason=_MW_SKIP)
def test_main_window_close_event_invokes_shutdown():
    """closeEvent must route through the central sweep (constructing a
    full MainWindow headlessly is out of scope for a unit test, so pin
    the call site)."""
    assert _method_calls(main_window.MainWindow.closeEvent,
                         '_shutdown_dock_workers') or \
        '_shutdown_dock_workers' in {
            n.func.id
            for n in ast.walk(ast.parse(textwrap.dedent(
                inspect.getsource(main_window.MainWindow.closeEvent))))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


# ═══════════════════════════════════════════════════════════════════
# P2-39: spot_field_dock visibility gate
# ═══════════════════════════════════════════════════════════════════

class _StubSpotModel(QObject):
    system_changed = Signal()
    wavelength_m = 550e-9
    epd_m = 5e-3
    field_angles_deg = [0.0]

    def build_trace_surfaces(self):
        return []          # -> _replot exits on the cheap message path


@pytest.fixture
def spot_dock():
    _app()
    sm = _StubSpotModel()
    dock = spot_field_dock.SpotFieldDock(sm)
    yield sm, dock
    dock.hide()


def _count_replots(dock, counter):
    orig = dock._replot

    def counted():
        counter.append(1)
        orig()
    dock._replot = counted


def test_spot_field_hidden_emit_marks_stale_without_replot(spot_dock):
    """system_changed while hidden must NOT run the full replot
    (pre-fix: 2 emits -> 2 full recomputes on the GUI thread), only
    mark the dock stale."""
    sm, dock = spot_dock
    assert not dock.isVisible()
    calls = []
    _count_replots(dock, calls)
    sm.system_changed.emit()
    sm.system_changed.emit()
    assert calls == []
    assert dock._stale is True


def test_spot_field_show_recomputes_once_when_stale(spot_dock):
    """showEvent must run exactly one catch-up replot iff stale."""
    sm, dock = spot_dock
    sm.system_changed.emit()             # hidden -> stale
    calls = []
    _count_replots(dock, calls)
    dock.show()
    assert len(calls) == 1
    assert dock._stale is False
    dock.hide()
    dock.show()                          # not stale -> no extra replot
    assert len(calls) == 1


def test_spot_field_visible_emit_replots_immediately(spot_dock):
    """While visible the legacy live-update behaviour is unchanged."""
    sm, dock = spot_dock
    dock.show()
    assert dock.isVisible()
    calls = []
    _count_replots(dock, calls)
    sm.system_changed.emit()
    assert len(calls) == 1
    assert dock._stale is False


# ═══════════════════════════════════════════════════════════════════
# P2-40: cooperative Stop (no QThread.terminate())
# ═══════════════════════════════════════════════════════════════════

def test_stop_paths_use_interruption_not_terminate():
    """Neither Stop slot may call .terminate() or manually invoke
    _on_finished (double-completion race) any more; both must request
    cooperative interruption."""
    for slot in (waveoptics_dock.WaveOpticsDock._stop,
                 through_focus_dock.ThroughFocusDock._stop_scan):
        assert not _method_calls(slot, 'terminate'), slot.__qualname__
        assert not _method_calls(slot, '_on_finished'), slot.__qualname__
        assert _method_calls(slot, 'requestInterruption'), slot.__qualname__


class _GatedTFWorker(through_focus_dock.ThroughFocusWorker):
    """Holds every progress-hook call at a gate so the test can request
    interruption deterministically mid-scan (cross-thread signal
    delivery to the test is queued, so a signal-driven request would
    race the scan)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = threading.Event()

    def _on_progress(self, stage, fraction, message=''):
        assert self.gate.wait(10), 'test gate never opened'
        super()._on_progress(stage, fraction, message)


def test_through_focus_worker_stops_between_planes():
    """requestInterruption mid-scan must abort between z-planes and
    emit the dock's established 'Stopped.' payload (None) -- pre-fix
    the request was ignored and the full result was emitted."""
    app = _app()
    E = np.ones((32, 32), dtype=np.complex128)
    z = np.linspace(-1e-3, 1e-3, 5)
    w = _GatedTFWorker(E, 1e-6, 633e-9, z)
    got = []
    w.finished_result.connect(got.append)
    w.start()
    try:
        deadline = time.time() + 5
        while not w.isRunning() and time.time() < deadline:
            time.sleep(0.01)
        w.requestInterruption()   # flag set BEFORE the hook may proceed
        w.gate.set()
        assert w.wait(10000)
    finally:
        _join(w)
    app.processEvents()           # deliver the queued finished_result
    assert got == [None]


def test_through_focus_worker_completes_without_interruption():
    """Control: the cooperative check must not break a normal run."""
    app = _app()
    E = np.ones((32, 32), dtype=np.complex128)
    z = np.linspace(-1e-3, 1e-3, 3)
    w = through_focus_dock.ThroughFocusWorker(E, 1e-6, 633e-9, z)
    got = []
    w.finished_result.connect(got.append)
    w.run()                       # synchronous: same-thread delivery
    app.processEvents()
    assert len(got) == 1
    assert got[0] is not None
    assert hasattr(got[0], 'z')   # ThroughFocusResult, not an error dict


class _FakeSurface:
    is_mirror = False
    is_coordbrk = False
    label = 'S'
    radius = np.inf
    conic = 0.0
    semi_diameter = np.inf
    thickness = 0.0
    glass_before = 'air'
    glass_after = 'air'


class _StubWaveModel:
    """Just enough SystemModel for WaveOpticsWorker.  Since the P3-63
    snapshot fix, build_trace_surfaces runs on the GUI thread inside
    __init__, so interruption is requested directly on the worker
    (before start) rather than from inside the run."""
    wavelength_m = 550e-9
    epd_m = 5e-3
    elements = []
    source = None
    bfl_mm = float('nan')

    def build_trace_surfaces(self):
        return [_FakeSurface(), _FakeSurface()]


def _wave_cfg():
    return {'N': 32, 'dx_m': 1e-6, 'method': 'asm', 'backend': 'numpy'}


def test_waveoptics_worker_stops_at_stage_boundary():
    """An interruption requested during the run must be honoured at the
    next stage boundary with the 'Stopped by user' payload -- the same
    payload the old terminate() path faked, so _on_finished handles it
    unchanged."""
    app = _app()
    from lumenairy.propagators import fft_infra
    orig = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT)
    model = _StubWaveModel()

    class _InterruptingCfg(dict):
        """Requests interruption from INSIDE the worker thread at the
        first cfg read after start (QThread.start() clears any flag
        requested beforehand; the P3-63 snapshot moved
        build_trace_surfaces -- the old in-thread hook -- to
        __init__ on the GUI thread)."""
        worker = None

        def get(self, key, default=None):
            if key == 'unfold_mirrors' and self.worker is not None:
                self.worker.requestInterruption()
            return super().get(key, default)

    cfg = _InterruptingCfg(_wave_cfg())
    w = waveoptics_dock.WaveOpticsWorker(model, cfg)
    cfg.worker = w
    got = []
    w.finished_result.connect(got.append)
    w.start()
    try:
        assert w.wait(10000)
    finally:
        _join(w)
        fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT = orig
    app.processEvents()
    assert got == [{'error': 'Stopped by user'}]


def test_waveoptics_worker_completes_without_interruption():
    """Control: with no interruption the stage checks must be inert and
    the run must finish with a results payload."""
    app = _app()
    from lumenairy.propagators import fft_infra
    orig = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT)
    try:
        w = waveoptics_dock.WaveOpticsWorker(_StubWaveModel(), _wave_cfg())
        got = []
        w.finished_result.connect(got.append)
        w.run()                   # synchronous, same-thread delivery
    finally:
        fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT = orig
    app.processEvents()
    assert len(got) == 1
    assert 'error' not in got[0]
    assert 'planes' in got[0]
