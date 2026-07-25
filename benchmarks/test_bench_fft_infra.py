"""pyFFTW infrastructure benchmarks for v4.12.

Measures the two perf changes against the v4.11.2 baseline:

* :func:`_fft2` / :func:`_ifft2` double-buffer (drops the per-call
  ``buf.copy()`` round-trip).
* Auto-promote of ESTIMATE -> MEASURE plans after
  ``_PYFFTW_AUTO_PROMOTE_THRESHOLD`` calls at the same key.

Two grid sizes are exercised: 1024x1024 (catches the per-call copy
elision) and 4096x4096 (catches the MEASURE promotion -- pocketfft
already saturates a single thread at this size, so the MEASURE plan
matters more here than the buffer copy).

Compare against ``v4_11_2`` by saving a baseline with
``pytest benchmarks/test_bench_fft_infra.py --benchmark-save=v4_12_0``
and then ``--benchmark-compare=v4_12_0`` from the next branch.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators import propagation as _prop

# ---------------------------------------------------------------------
# ASM at 1024^2 -- pyFFTW path
# ---------------------------------------------------------------------
# The H cache is pre-warmed so the benchmark measures the steady-state
# FFT pair, not the chunked H construction.  ``angular_spectrum_propagate``
# is the canonical hot path for the double-buffer change: the call
# does one ``_fft2`` and one ``_ifft2`` per propagation, each of which
# would have done a 16 MB ``buf.copy()`` under v4.11.2.

def _make_plane_wave(N: int, dtype=np.complex128) -> np.ndarray:
    """Tiny constant field -- the FFT cost dominates the call so
    we don't want the H build or input prep to add variance."""
    return np.ones((N, N), dtype=dtype)


def test_bench_asm_propagate_1024(benchmark):
    """ASM 1024x1024 complex128, steady-state (H cache pre-warmed,
    plan cache pre-warmed)."""
    N, dx, wl = 1024, 5e-6, 1.55e-6
    z = 1e-3
    E = _make_plane_wave(N)
    # Warm both caches: one call builds H + plans.  The second call
    # benefits from the cache but might still be inside the
    # auto-promote ramp; do a few warm-ups so the steady-state
    # benchmark is on the MEASURE plan if auto-promote is on.
    for _ in range(8):
        _ = lm.angular_spectrum_propagate(E, z, wl, dx)
    benchmark(lm.angular_spectrum_propagate, E, z, wl, dx)


def test_bench_asm_propagate_4096(benchmark):
    """ASM 4096x4096 complex128 -- the regime where MEASURE matters
    most.  Single iteration is ~200 ms on a decent multi-core box.

    Skipped on memory-tight hosts (need ~512 MB just for E_in + H +
    plan buffers); a 16 GB laptop runs it fine.
    """
    N, dx, wl = 4096, 5e-6, 1.55e-6
    z = 1e-3
    E = _make_plane_wave(N)
    # Pre-warm including the auto-promote (needs >= threshold calls).
    for _ in range(8):
        _ = lm.angular_spectrum_propagate(E, z, wl, dx)
    # Set a generous max_time so a single 4k FFT iteration counts.
    benchmark.pedantic(lm.angular_spectrum_propagate, args=(E, z, wl, dx),
                       rounds=5, iterations=1, warmup_rounds=0)


# ---------------------------------------------------------------------
# Direct _fft2 path -- isolates the wrapper overhead from the
# surrounding ASM kernel.  Useful for pinpointing whether a
# regression is in the FFT wrapper or in the H-cache + multiply.
# ---------------------------------------------------------------------

def test_bench_fft2_1024(benchmark):
    """Bare ``_fft2`` on a 1024x1024 complex128 grid."""
    N = 1024
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    # Warm so the plan is promoted to MEASURE if auto-promote is on.
    for _ in range(8):
        _ = _prop._fft2(x)
    benchmark(_prop._fft2, x)


def test_bench_ifft2_1024(benchmark):
    """Bare ``_ifft2`` on a 1024x1024 complex128 grid."""
    N = 1024
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    for _ in range(8):
        _ = _prop._ifft2(x)
    benchmark(_prop._ifft2, x)


# ---------------------------------------------------------------------
# Auto-promote-on vs auto-promote-off comparison (4096^2).
# ---------------------------------------------------------------------
# Two paired benchmarks so a developer can eyeball the ESTIMATE-only
# steady-state cost vs the MEASURE-promoted steady-state cost.  Skip
# on lower-memory hosts as the 4k plan rebuild is expensive in time
# even when memory allows it.

@pytest.mark.parametrize('auto_promote', [True, False],
                          ids=['promote_on', 'promote_off'])
def test_bench_fft2_1024_promote_compare(benchmark, auto_promote):
    """Compare _fft2 at 1024^2 with auto-promote on vs off.

    Each parametrisation resets the plan cache so the steady-state
    measurement reflects the active promotion policy.
    """
    _prop.reset_fft_backend()
    prev = _prop.get_fft_auto_promote()
    _prop.set_fft_auto_promote(bool(auto_promote))
    try:
        N = 1024
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        # Warm enough that any pending auto-promote has fired.
        for _ in range(8):
            _ = _prop._fft2(x)
        benchmark(_prop._fft2, x)
    finally:
        _prop.set_fft_auto_promote(prev)
        _prop.reset_fft_backend()
