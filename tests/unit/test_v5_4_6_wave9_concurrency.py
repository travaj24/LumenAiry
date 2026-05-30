"""v5.4.6 Wave 9 regression pins: FFT/storage/concurrency cluster.

- P3-16: snapshot_fft_state / restore_fft_state round-trip the dispatch
         globals (so spawn workers can inherit parent config).
- F-32 : warmup_fft_plans defaults threads to the FFTW_THREADS global.

P3-13 (MEASURE-under-lock) is a documented perf limitation; P3-14
(bad-shapes race) and P3-15 (HDF5 orphan rollback) are surgical
concurrency / fault-path fixes covered by the storage/fft suite remaining
green.
"""
from __future__ import annotations

import inspect


def test_snapshot_restore_fft_state_round_trip():
    import lumenairy.propagators.fft_infra as fi
    from lumenairy.propagators.fft_infra import (
        restore_fft_state,
        snapshot_fft_state,
    )
    state = snapshot_fft_state()
    assert 'FFTW_THREADS' in state and 'DEFAULT_COMPLEX_DTYPE' in state
    orig = fi.FFTW_THREADS
    try:
        fi.FFTW_THREADS = 12345  # simulate a worker at the wrong value
        restore_fft_state(state)
        assert fi.FFTW_THREADS == state['FFTW_THREADS']
    finally:
        fi.FFTW_THREADS = orig
    # forward-compatible: ignores unknown / missing keys
    restore_fft_state({'BOGUS_KEY': 1})
    restore_fft_state({})


def test_warmup_defaults_to_fftw_threads_global():
    """warmup_fft_plans must default the thread count to the FFTW_THREADS
    global that _fft2/_ifft2 dispatch on (F-32) -- not _available_cpus()."""
    import lumenairy.propagators.fft_infra as fi
    src = inspect.getsource(fi.warmup_fft_plans)
    assert 'FFTW_THREADS' in src, (
        "warmup_fft_plans must reference the FFTW_THREADS dispatch global")
