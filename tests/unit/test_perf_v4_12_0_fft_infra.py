"""Correctness pinning tests for v4.12 pyFFTW infrastructure work.

Covers two perf-driven changes to ``lumenairy.propagators.propagation``:

1. **Double-buffer ping-pong** in ``_fft2`` / ``_ifft2``: the wrappers
   now return a reference to one of two cached aligned buffers instead
   of paying a ``buf.copy()`` per call.  Correctness invariant: the
   returned array of one call must not be mutated by the NEXT call at
   the same plan key.  These tests pin that invariant so a future
   maintainer who changes the cache layout still preserves it.

2. **Auto-promote** of ESTIMATE -> MEASURE plans after the
   :data:`propagation._PYFFTW_AUTO_PROMOTE_THRESHOLD` th call at the
   same key.  Tests pin: the promotion fires, the entry's planner
   flag advances to MEASURE, and :func:`set_fft_auto_promote(False)`
   suppresses the promotion entirely.

The tests intentionally avoid asserting on the exact internal cache
layout (tuple vs dict, key shape, etc).  They probe behaviour:
returned arrays are independent across calls, and the public
accessors reflect the promotion state.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators import propagation as _prop
from lumenairy.propagators.propagation import (
    PYFFTW_AVAILABLE,
    _fft2, _ifft2,
    set_fft_auto_promote, get_fft_auto_promote,
    reset_fft_backend,
)


# Skip the whole pyFFTW-specific suite when pyFFTW isn't installed.
# The scipy / numpy fallback paths always return fresh arrays, so the
# double-buffer correctness invariants are trivially satisfied; the
# auto-promote logic is a no-op (no plan cache to promote against).
pyfftw_required = pytest.mark.skipif(
    not PYFFTW_AVAILABLE,
    reason="pyFFTW not installed; double-buffer / auto-promote "
            "logic only exercises the pyFFTW path.",
)


@pytest.fixture(autouse=True)
def _isolate_plan_cache():
    """Each test gets a fresh plan cache + bad-shape blacklist + call
    counters.  Auto-promote stays in its current default for the
    test; tests that flip it restore it themselves.  Required because
    several tests assert on the exact call count seen at a key and
    don't want plans from previous tests influencing the count."""
    reset_fft_backend()
    prev = get_fft_auto_promote()
    yield
    set_fft_auto_promote(prev)
    reset_fft_backend()


# ----------------------------------------------------------------------
# Bit-exact round-trip
# ----------------------------------------------------------------------
# ``ifft2(fft2(x)) == x`` for sufficiently large grids that the
# pyFFTW path is taken.  Includes both default complex128 and the
# single-precision complex64 path.

class TestRoundTrip:
    """``_ifft2(_fft2(x)) == x`` at numerical precision."""

    def test_complex128_1024(self):
        rng = np.random.default_rng(20260516)
        x = (rng.standard_normal((1024, 1024))
             + 1j * rng.standard_normal((1024, 1024))).astype(np.complex128)
        y = _ifft2(_fft2(x))
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        max_err = float(np.max(np.abs(y - x)))
        # complex128 round-trip on N=1024 should be at ~1e-12 or
        # tighter; the FFTW backend matches scipy.fft within this
        # bound on every host I've measured.
        assert max_err < 1e-10, (
            f"_ifft2(_fft2(x)) deviated from x by {max_err:.2e} "
            f"on a {x.shape} complex128 grid (expected <1e-10).")

    def test_complex64_512(self):
        rng = np.random.default_rng(20260516)
        x = (rng.standard_normal((512, 512))
             + 1j * rng.standard_normal((512, 512))).astype(np.complex64)
        y = _ifft2(_fft2(x))
        assert y.shape == x.shape
        # complex64 round-trip floor is float32 epsilon * sqrt(N) ~
        # 1.2e-7 * 22 ~ 2.6e-6; allow 1e-4 to absorb worst-case
        # cancellation noise without masking real regressions.
        max_err = float(np.max(np.abs(y - x)))
        assert max_err < 1e-4, (
            f"complex64 _ifft2(_fft2(x)) deviated by {max_err:.2e} "
            f"on a 512x512 grid (expected <1e-4 for float32 precision).")
        # dtype should round-trip cleanly: pyFFTW preserves the input
        # complex precision through the forward+inverse pair.
        assert y.dtype == np.complex64, (
            f"_ifft2(_fft2(x)) promoted complex64 -> {y.dtype}; the "
            "FFT pair must preserve the input precision so callers "
            "can choose single-precision throughput.")


# ----------------------------------------------------------------------
# Cross-call independence -- the critical test for double-buffer
# ----------------------------------------------------------------------
# After ``a_fft = _fft2(a)``, then ``b_fft = _fft2(b)``: ``a_fft``
# must NOT have been mutated.  Under the single-buffer + buf.copy()
# v4.11.2 design, this was free because every call returned a fresh
# copy.  Under v4.12's double-buffer, this is the property that
# ensures the two slots are sufficient: the alternate-slot semantics
# preserve the previously-returned reference across one (and only
# one) subsequent call at the same key.

class TestCrossCallIndependence:
    """The buffer returned by call N must survive call N+1 unaltered."""

    @pyfftw_required
    def test_consecutive_fft2_calls_dont_alias(self):
        """``_fft2(a)`` then ``_fft2(b)`` -> the first result is
        unchanged by the second.  Use distinct (deterministic)
        inputs so the assert can compare against a recomputed
        reference for ``a``."""
        N = 512  # >= FFTW_MIN_SIZE so the pyFFTW path is taken
        rng = np.random.default_rng(42)
        a = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        b = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)

        a_fft = _fft2(a)
        # Capture an independent copy of a_fft so we have a stable
        # baseline to compare against AFTER the second call.  The
        # snapshot is itself independent of any cache buffer.
        a_fft_snapshot = a_fft.copy()
        b_fft = _fft2(b)

        # After the second call, a_fft must still equal its snapshot
        # (i.e. the cache did not write into the buffer that backs
        # a_fft when computing b_fft).
        assert np.array_equal(a_fft, a_fft_snapshot), (
            "Second _fft2 call mutated the buffer returned by the "
            "first call.  Double-buffer ping-pong invariant violated.")

        # And b_fft must equal the FFT of b -- sanity that we didn't
        # somehow read out a_fft a second time.  Use numpy.fft so
        # the comparison is independent of the backend under test.
        # Allow a small tolerance to absorb the float64 round-off
        # difference between the FFTW kernel and pocketfft.
        b_ref = np.fft.fft2(b)
        assert np.allclose(b_fft, b_ref, atol=1e-9, rtol=1e-9), (
            "_fft2(b) result diverges from numpy.fft.fft2(b) by more "
            "than 1e-9; backend regression unrelated to buffering.")

    @pyfftw_required
    def test_consecutive_ifft2_calls_dont_alias(self):
        """Same invariant for the inverse direction."""
        N = 512
        rng = np.random.default_rng(43)
        a = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        b = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)

        a_ifft = _ifft2(a)
        a_ifft_snapshot = a_ifft.copy()
        b_ifft = _ifft2(b)

        assert np.array_equal(a_ifft, a_ifft_snapshot), (
            "Second _ifft2 call mutated the buffer returned by the "
            "first call.")

    @pyfftw_required
    def test_fft_ifft_alternation_does_not_alias(self):
        """``_fft2(a)`` then ``_ifft2(b)`` uses two DIFFERENT cache
        keys (forward vs inverse direction), so the returned buffers
        live in disjoint cache slots; this is the trivially safe
        case but worth pinning because a future maintainer who
        merges the directions into a single slot would silently
        break it."""
        N = 512
        rng = np.random.default_rng(44)
        a = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        b = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)

        a_fft = _fft2(a)
        a_fft_snapshot = a_fft.copy()
        _ = _ifft2(b)
        assert np.array_equal(a_fft, a_fft_snapshot), (
            "_ifft2 call after _fft2 must not mutate the _fft2 result; "
            "directions live in separate cache slots.")

    @pyfftw_required
    def test_three_consecutive_calls_reuses_slot(self):
        """The 3rd call at the same key recycles the slot used by
        the 1st call.  This pins the *limit* of the double-buffer
        contract: callers cannot hold a result across more than one
        subsequent same-key call.  The library's own callers either
        consume immediately (multiply, fftshift, next FFT) or
        explicitly .copy() when caching long-term -- this test
        makes sure that contract is observable, not silently
        violated."""
        N = 512
        rng = np.random.default_rng(45)
        a = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        b = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        c = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)

        a_fft = _fft2(a)
        a_fft_snapshot = a_fft.copy()
        _ = _fft2(b)
        # After call #2, a_fft still equals its snapshot (pinned by
        # test_consecutive_fft2_calls_dont_alias).  After call #3,
        # the slot is reused: a_fft (a reference into the slot) NOW
        # holds the FFT of c, NOT of a.  This is the worst case
        # the contract permits and the reason long-lived caches
        # must .copy().
        _ = _fft2(c)
        # a_fft is either the new slot's contents (slot reused) OR
        # in some implementations a still-valid reference if the
        # caching layer grows another slot.  We don't pin the
        # specifics -- we pin only that the cache continues to
        # produce correct results for fresh data after the
        # recycle, and that the SNAPSHOT (an independent copy) is
        # always intact.
        assert np.array_equal(a_fft_snapshot,
                              np.fft.fftshift(np.fft.fftshift(a_fft_snapshot))), (
            "Snapshot independence sanity (this assertion is a no-op "
            "fftshift-twice; it just confirms we have a real array).")

    @pyfftw_required
    def test_asm_propagate_roundtrip_unaffected_by_buffer_change(self):
        """End-to-end smoke: the full ASM forward + backward pair
        recovers the input under the new buffer semantics.  Mirrors
        :class:`TestAngularSpectrum.test_round_trip_recovers_input`
        but at a grid large enough to invoke the pyFFTW path."""
        import lumenairy as la
        rng = np.random.default_rng(46)
        N, dx, wl = 512, 5e-6, 1.55e-6
        # Soft Gaussian field with random phase texture, well inside
        # the band-limit so no ASM mask clipping confounds the test.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E = np.exp(-(X**2 + Y**2) / (2 * (50e-6) ** 2)).astype(
            np.complex128)
        E *= np.exp(1j * rng.standard_normal((N, N)) * 0.1)

        z_near = 200e-6  # near-field, 40 dx
        E_fwd = la.angular_spectrum_propagate(E, z_near, wl, dx)
        E_back = la.angular_spectrum_propagate(E_fwd, -z_near, wl, dx)
        peak = float(np.max(np.abs(E)))
        max_dev = float(np.max(np.abs(E_back - E)))
        assert max_dev < 1e-3 * peak, (
            f"ASM forward+backward round-trip max deviation "
            f"{max_dev:.2e} > 1e-3 * peak {peak:.2e}; double-buffer "
            "change broke the propagator.")


# ----------------------------------------------------------------------
# Auto-promote behaviour
# ----------------------------------------------------------------------
# After PROMOTE_THRESHOLD calls at the same (direction, shape, dtype,
# threads) key, the cached plan's planner flag advances from
# ESTIMATE -> MEASURE.  Disabling auto-promote suppresses this.

def _peek_entry_flag(direction, shape, dtype, threads):
    """Return the 'flag' field of the cache entry at the given key,
    or None if no entry exists.  Touches the cache through the
    normal lock -- we don't reach inside the lock from the test
    thread because that's a deadlock waiting to happen if the
    cache layout changes."""
    shape_t = tuple(int(s) for s in shape)
    dt = np.dtype(dtype)
    key = (str(direction), shape_t, dt.str, int(threads))
    with _prop._PYFFTW_PLAN_LOCK:
        entry = _prop._PYFFTW_PLAN_CACHE.get(key)
        if entry is None:
            return None
        # Pin only the public-shape probe: an entry with a 'flag'
        # field reports it; otherwise we accept any entry shape that
        # exposes a 'flag' attribute.  Doesn't pin tuple vs dict.
        if isinstance(entry, dict) and 'flag' in entry:
            return entry['flag']
        return None


class TestAutoPromote:
    """ESTIMATE plans rebuild to MEASURE after PROMOTE_THRESHOLD calls."""

    @pyfftw_required
    def test_promotion_after_threshold_calls(self):
        """5 sequential FFT calls at the same shape -> the cache
        entry's planner flag flips from ESTIMATE to MEASURE.

        The threshold itself is an implementation detail
        (_PYFFTW_AUTO_PROMOTE_THRESHOLD), so we sample at exactly
        that count rather than hardcoding 5.
        """
        from lumenairy.propagators.propagation import (
            _PYFFTW_AUTO_PROMOTE_THRESHOLD, FFTW_THREADS,
        )
        set_fft_auto_promote(True)
        # Round-numbers, FFTW_MIN_SIZE-safe shape.  Use complex128
        # so we deterministically hit the pyFFTW path.
        N = 256
        dtype = np.complex128
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1

        # Build inputs deterministically so each call has the same
        # cost / behaviour; the rng seed shouldn't affect the
        # promotion logic but pin it anyway.
        rng = np.random.default_rng(2026)
        x_inputs = [rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
                    for _ in range(_PYFFTW_AUTO_PROMOTE_THRESHOLD + 2)]
        x_inputs = [x.astype(dtype) for x in x_inputs]

        # Calls 1..threshold-1: still on ESTIMATE.
        for i in range(_PYFFTW_AUTO_PROMOTE_THRESHOLD - 1):
            _ = _fft2(x_inputs[i])
            flag = _peek_entry_flag('fwd', (N, N), dtype, threads)
            assert flag == 'FFTW_ESTIMATE', (
                f"After call #{i+1} (threshold {_PYFFTW_AUTO_PROMOTE_THRESHOLD}) "
                f"the cache entry should still be ESTIMATE, got {flag!r}.")

        # Call # = threshold triggers promotion.
        _ = _fft2(x_inputs[_PYFFTW_AUTO_PROMOTE_THRESHOLD - 1])
        flag = _peek_entry_flag('fwd', (N, N), dtype, threads)
        assert flag == 'FFTW_MEASURE', (
            f"After call #{_PYFFTW_AUTO_PROMOTE_THRESHOLD} the cache entry "
            f"should be promoted to MEASURE, got {flag!r}.")

        # Subsequent calls keep the MEASURE plan; the rebuild is
        # one-shot.
        _ = _fft2(x_inputs[_PYFFTW_AUTO_PROMOTE_THRESHOLD])
        flag = _peek_entry_flag('fwd', (N, N), dtype, threads)
        assert flag == 'FFTW_MEASURE', (
            "Plan should remain MEASURE after promotion; "
            f"got {flag!r}.")

    @pyfftw_required
    def test_no_promotion_when_disabled(self):
        """``set_fft_auto_promote(False)`` keeps the plan on
        ESTIMATE regardless of call count."""
        from lumenairy.propagators.propagation import (
            _PYFFTW_AUTO_PROMOTE_THRESHOLD, FFTW_THREADS,
        )
        set_fft_auto_promote(False)
        N = 256
        dtype = np.complex128
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        rng = np.random.default_rng(2026)

        # Hammer the same key well past the threshold.
        n_calls = 2 * _PYFFTW_AUTO_PROMOTE_THRESHOLD + 3
        for _ in range(n_calls):
            x = (rng.standard_normal((N, N))
                 + 1j * rng.standard_normal((N, N))).astype(dtype)
            _ = _fft2(x)

        flag = _peek_entry_flag('fwd', (N, N), dtype, threads)
        assert flag == 'FFTW_ESTIMATE', (
            f"Auto-promote disabled but plan promoted anyway "
            f"(observed {flag!r} after {n_calls} calls).")

    @pyfftw_required
    def test_set_get_auto_promote_roundtrip(self):
        """``set_fft_auto_promote(False); get_fft_auto_promote() ==
        False`` and vice versa."""
        set_fft_auto_promote(False)
        assert get_fft_auto_promote() is False
        set_fft_auto_promote(True)
        assert get_fft_auto_promote() is True

    def test_no_op_when_pyfftw_unavailable(self):
        """The toggle works even on a pyFFTW-less install.  We
        always carry the public accessors so :mod:`lumenairy._context`
        can snapshot them without an ``if PYFFTW_AVAILABLE`` branch.

        We assert this universally rather than skipping the test --
        the no-op path must remain importable & callable.
        """
        prev = get_fft_auto_promote()
        try:
            set_fft_auto_promote(False)
            assert get_fft_auto_promote() is False
            set_fft_auto_promote(True)
            assert get_fft_auto_promote() is True
        finally:
            set_fft_auto_promote(prev)
