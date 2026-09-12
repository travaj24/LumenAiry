"""VERIFY-A5 -- the pyFFTW ping-pong buffer must not be shared across threads.

Found while re-verifying WP-A5's K4 (one lock per plan slot instead of one per
entry).  The double-buffer contract documented on ``_fft2`` -- "your slot stays
valid until the call AFTER next" -- is a SINGLE-THREADED statement.  With T
threads issuing FFTs at the same ``(direction, shape, dtype, threads)`` key in
arbitrary order, a caller's slot can be recycled while it still holds the
reference; the caller then multiplies a stale or half-written spectrum and
returns a silently 100 %-wrong field.

The entry-wide lock that K4 replaced had been masking this by serialising every
call at a key.  MEASURED on this box (8 threads x 40 concurrent
``rayleigh_sommerfeld_propagate(128x128, complex128, z = 5 mm)`` calls, each
compared byte-for-byte with the single-threaded reference):

    per-slot locks as landed by K4 : 7 / 320 wrong, max|out-ref|/max|ref| = 1.14
    entry-wide lock (pre-K4)       : 0 / 320
    single buffer (double off)     : 0 / 320
    per-slot locks + this fix      : 0 / 320
    ``angular_spectrum_propagate`` : 0 / 320 in every configuration

Fix: ``fft_infra._note_fft_thread`` latches the moment a SECOND thread issues an
FFT, after which every pyFFTW return privatises its buffer.  Single-threaded
callers keep the zero-copy path bit-identically.

Author: VERIFY-A5.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np

import lumenairy.propagators.fft_infra as _fi
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.fft_infra import _fft2, clear_asm_caches
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

LAM = 633e-9


def _probe_field(N=128, dx=1e-6):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / (8e-6) ** 2).astype(np.complex128), dx


class TestFftBuffersAreNotSharedAcrossThreads:

    def test_concurrent_propagations_match_the_single_threaded_reference(self):
        """Bit-identity, so there is no tolerance to derive: the same
        deterministic call from N threads must give the same array it gives
        from one.

        Power of the pin: the pre-fix failure rate measured 7 / 320 at
        8 threads x 40 reps; this runs 6 x 20 = 120 concurrent calls, i.e.
        an expected ~2.6 failures pre-fix.  It cannot fail post-fix at any
        rate -- once a second thread is seen, no caller is handed a shared
        buffer at all, so the property is structural, not statistical.
        """
        E, dx = _probe_field()
        z = 5e-3
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref_rs = np.asarray(
                rayleigh_sommerfeld_propagate(E.copy(), z, LAM, dx))
            ref_asm = np.asarray(
                angular_spectrum_propagate(E.copy(), z, LAM, dx))
        bad = []

        def worker():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(20):
                    a = np.asarray(
                        rayleigh_sommerfeld_propagate(E.copy(), z, LAM, dx))
                    if not np.array_equal(a, ref_rs):
                        bad.append(('rs', float(np.max(np.abs(a - ref_rs))
                                                / np.max(np.abs(ref_rs)))))
                    b = np.asarray(
                        angular_spectrum_propagate(E.copy(), z, LAM, dx))
                    if not np.array_equal(b, ref_asm):
                        bad.append(('asm', float(np.max(np.abs(b - ref_asm))
                                                 / np.max(np.abs(ref_asm)))))

        ths = [threading.Thread(target=worker) for _ in range(6)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        assert not bad, (
            f"{len(bad)} of 240 concurrent propagations disagreed with the "
            f"single-threaded reference (worst relative deviation "
            f"{max(d for _k, d in bad):.3f}); the pyFFTW ping-pong slot was "
            f"recycled under a caller that still held it.")

    def test_the_buffer_is_privatised_once_a_second_thread_uses_the_fft(self):
        """Structural counterpart to the statistical pin above, and the
        counter-pin that the single-threaded zero-copy path is untouched.

        Both halves are decisions (does the returned array alias the plan
        buffer?), not readings, so neither needs a numeric bar.
        """
        saved = (_fi._PYFFTW_FIRST_FFT_THREAD, _fi._PYFFTW_SHARED_BUFFERS_UNSAFE)
        try:
            _fi._PYFFTW_FIRST_FFT_THREAD = None
            _fi._PYFFTW_SHARED_BUFFERS_UNSAFE = False
            clear_asm_caches()
            rng = np.random.default_rng(11)
            a = (rng.standard_normal((256, 256))
                 + 1j * rng.standard_normal((256, 256))).astype(np.complex128)
            b = a * 2.0
            c = a * 3.0

            first = _fft2(a)
            second = _fft2(b)
            pyfftw_in_play = (getattr(_fi, 'PYFFTW_AVAILABLE', False)
                              and getattr(_fi, 'USE_PYFFTW', False)
                              and getattr(_fi, '_PYFFTW_DOUBLE_BUFFER', False))
            if pyfftw_in_play:
                # single-threaded: the third call recycles the first slot,
                # which is exactly the zero-copy behaviour the double buffer
                # exists for and which this fix must NOT change.
                third = _fft2(c)
                assert np.shares_memory(first, third), (
                    "single-threaded zero-copy ping-pong regressed: the 3rd "
                    "call no longer reuses the 1st call's slot.")
            del first, second

            def other_thread():
                _fft2(a)

            t = threading.Thread(target=other_thread)
            t.start()
            t.join()
            assert _fi._PYFFTW_SHARED_BUFFERS_UNSAFE is True, (
                "a second thread issued an FFT but the shared-buffer latch "
                "did not trip.")
            p = _fft2(a)
            q = _fft2(b)
            r = _fft2(c)
            assert not np.shares_memory(p, r), (
                "after multi-threaded use the pyFFTW ping-pong slot is still "
                "handed out as a view; a concurrent caller can have its "
                "spectrum overwritten mid-use.")
            assert not np.shares_memory(p, q)
            assert np.allclose(r, np.fft.fft2(c), rtol=1e-10, atol=1e-8), (
                "privatised buffers must still carry the right transform.")
        finally:
            _fi._PYFFTW_FIRST_FFT_THREAD, _fi._PYFFTW_SHARED_BUFFERS_UNSAFE = saved
            clear_asm_caches()
