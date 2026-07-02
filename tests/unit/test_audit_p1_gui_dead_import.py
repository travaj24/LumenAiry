"""Audit P1-08 (v5.17.0 deep audit): WaveOpticsWorker.run imported the
nonexistent pre-v5.14-reorg module ``lumenairy.propagation`` to set the
FFT backend flags.  The import sat BEFORE the big try in run(), so every
GUI wave-optics run died with ModuleNotFoundError, the custom finished
signal was never emitted, and the dock hung at 'Running...' forever.

Fix under test (lumenairy/ui/waveoptics_dock.py):
  (a) backend selection now sets USE_PYFFTW / USE_SCIPY_FFT on
      ``lumenairy.propagators.fft_infra`` itself (where _fft2/_ifft2
      read their module globals); and
  (b) run() is a failure-safe wrapper around _run_impl() so ANY escaping
      exception still emits finished with an {'error': ...} payload.

Pre-fix, the two worker tests below fail: run() raises
ModuleNotFoundError out of the thread body and the finished collector
stays empty (verified via git-stash A/B).
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

# Probe GUI deps: on CI the [gui] extra is not installed and we skip
# the worker tests cleanly.  Locally everything imports and runs.
try:
    import matplotlib  # noqa: F401
    from PySide6.QtWidgets import QApplication
    matplotlib.use('Agg')
    from lumenairy.ui.waveoptics_dock import WaveOpticsWorker
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'


def test_dead_module_path_is_gone_and_fft_infra_flags_exist():
    """The import target the dock uses must exist; the pre-reorg path
    must not silently reappear (a shim would mask future regressions of
    this kind rather than fix them)."""
    from lumenairy.propagators import fft_infra
    assert isinstance(fft_infra.USE_PYFFTW, bool)
    assert isinstance(fft_infra.USE_SCIPY_FFT, bool)
    with pytest.raises(ModuleNotFoundError):
        import lumenairy.propagation  # noqa: F401


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
    """Minimal stand-in for SystemModel: enough attributes for the
    P3-63 __init__ snapshot and the pre-try section of
    WaveOpticsWorker._run_impl."""
    wavelength_m = 550e-9
    epd_m = 5e-3
    elements = []
    source = None

    def build_trace_surfaces(self):
        return [_FakeSurface()]


class _ProbeCfg(dict):
    """Config dict that records the live fft_infra backend flags and
    aborts the run at the first cfg read AFTER backend selection
    ('unfold_mirrors', still before _run_impl's internal try).  Since
    the P3-63 snapshot fix the model is no longer touched inside
    run(), so the cfg is the natural in-run probe point."""

    def __init__(self, *a, exc=None, **kw):
        super().__init__(*a, **kw)
        self.seen_flags = None
        self.exc = exc or RuntimeError('stub abort')

    def get(self, key, default=None):
        if key == 'unfold_mirrors':
            from lumenairy.propagators import fft_infra
            self.seen_flags = (fft_infra.USE_PYFFTW,
                               fft_infra.USE_SCIPY_FFT)
            raise self.exc
        return super().get(key, default)


def _run_worker(backend, exc=None):
    """Run a worker synchronously (plain run() call, direct-connection
    signals, no event loop) and return (probe_cfg, finished_payloads)."""
    QApplication.instance() or QApplication([])
    from lumenairy.propagators import fft_infra
    orig = (fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT)
    try:
        cfg = _ProbeCfg({'N': 32, 'dx_m': 1e-6, 'method': 'asm',
                         'backend': backend}, exc=exc)
        worker = WaveOpticsWorker(_StubModel(), cfg)
        got = []
        worker.finished_result.connect(got.append)
        worker.run()  # must NOT raise (P1-08)
        return cfg, got
    finally:
        # run() mutates process-global backend flags by design; restore
        # so later tests see the environment default.
        fft_infra.USE_PYFFTW, fft_infra.USE_SCIPY_FFT = orig


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_worker_run_emits_finished_on_early_exception():
    """Any exception escaping the body -- including ones raised BEFORE
    the internal try (here: the cfg probe) -- must still emit
    finished_result with an error payload so the dock can re-enable
    Run."""
    _, got = _run_worker('numpy')
    assert len(got) == 1
    assert 'error' in got[0]
    assert 'stub abort' in got[0]['error']


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
@pytest.mark.parametrize('backend, expect', [
    ('numpy', (False, False)),
    ('scipy', (False, True)),
    ('pyfftw', (True, False)),
])
def test_worker_backend_selection_reaches_fft_infra(backend, expect):
    """The combo-box backend must land on fft_infra's own module
    globals (the ones _fft2/_ifft2 read), not on a facade module."""
    cfg, got = _run_worker(backend)
    assert cfg.seen_flags == expect
    assert len(got) == 1 and 'error' in got[0]


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_worker_run_survives_exception_with_broken_str():
    """Wave-1 adversarial follow-up: an exception whose __str__ itself raises
    must STILL emit finished (run() formats the message defensively; a broken
    __str__ would otherwise escape and re-create the stuck-at-Running hang)."""
    class _EvilError(Exception):
        def __str__(self):
            raise ValueError('__str__ is broken')

    _, got = _run_worker('numpy', exc=_EvilError())
    assert len(got) == 1
    assert got[0]['error'] == '_EvilError'
