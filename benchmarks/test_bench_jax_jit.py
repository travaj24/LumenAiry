"""Benchmark the v4.12 JAX-jit perf work.

For each of the jit-cached entry points we measure:

* **First-call** cost (trace + compile + execute), captured manually
  before pytest-benchmark wraps the call.
* **Steady-state** cost (10th call onwards) captured via the
  ``benchmark`` fixture.

We then attach ``first_call_ms`` to the benchmark's ``extra_info``
dict so ``pytest-benchmark`` reports both numbers in the JSON output.
Expected speedup on the steady-state side: 5-50x vs the first-call
cost (workload-dependent; large grids see the biggest win).

If JAX is not installed in the runtime environment, the whole file
skips via ``pytest.importorskip``.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

# Skip the whole benchmark file if JAX is unavailable.
jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


import lumenairy as lm  # noqa: E402


def _time_one(fn, *args, **kwargs):
    """Run ``fn`` once and return wall-clock seconds.  Uses
    ``block_until_ready`` so async JAX kernels are timed properly."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    # JAX arrays have block_until_ready; tuples/namedtuples may not.
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


# ===========================================================================
# trace_jax -- restored in v4.12.1 (cache; benchmark in v4.12.2)
# ===========================================================================

def test_bench_trace_jax_first_vs_warm(benchmark):
    """``trace_jax`` first call (trace+compile) vs 10th call (cached
    kernel reuse).  v4.12.1 ships the pytree-registered
    :class:`JaxPrescription` cache; v4.12.2 reconciles the headline
    claim by re-running this benchmark and replacing the placeholder
    timings in the release notes.
    """
    from lumenairy.raytrace.jax_trace import (
        _TRACE_JAX_CACHE,
        make_jax_ray_state,
        trace_jax,
    )

    # Standard 1001-ray fan into a BK7 singlet (matches the test
    # fixture in test_audit_fixes_v4_12_1_trace_jax_cache.py).
    singlet = lm.make_singlet(
        R1=20e-3, R2=float('inf'), d=2e-3,
        glass='N-BK7', aperture=4e-3)
    rng = np.random.default_rng(0)
    n_rays = 1001
    rs = np.sqrt(rng.uniform(0, 1, n_rays)) * 2e-3
    theta = rng.uniform(0, 2 * np.pi, n_rays)
    x = jnp.asarray(rs * np.cos(theta))
    y = jnp.asarray(rs * np.sin(theta))
    state = make_jax_ray_state(
        x=x, y=y, z=jnp.zeros(n_rays),
        L=jnp.zeros(n_rays), M=jnp.zeros(n_rays),
        N=jnp.ones(n_rays))
    wl = 633e-9

    _TRACE_JAX_CACHE.clear()
    first_s = _time_one(trace_jax, state, singlet, wl)
    for _ in range(9):
        out = trace_jax(state, singlet, wl)
        out.x.block_until_ready()
    benchmark(lambda: trace_jax(state, singlet, wl).x.block_until_ready())
    benchmark.extra_info['first_call_ms'] = first_s * 1e3
    benchmark.extra_info['speedup_first_to_warm'] = (
        first_s / benchmark.stats.stats.mean
        if benchmark.stats.stats.mean > 0 else float('inf'))


# ===========================================================================
# propagate_through_system_jax
# ===========================================================================

def test_bench_propagate_through_system_jax_first_vs_warm(benchmark):
    """``propagate_through_system_jax`` first call vs 10th call."""
    from lumenairy.system import (
        propagate_through_system_jax,
        _PROPAGATE_SYSTEM_JAX_CACHE,
    )
    N, dx, wl = 128, 5e-6, 1.55e-6
    E = np.ones((N, N), dtype=np.complex64)
    elements = [
        {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
        {'type': 'lens', 'f': 5e-3},
        {'type': 'aperture', 'shape': 'circular',
         'params': {'radius': 200e-6}},
        {'type': 'propagate', 'z': 5e-3, 'bandlimit': True},
    ]

    _PROPAGATE_SYSTEM_JAX_CACHE.clear()
    first_s = _time_one(propagate_through_system_jax, E, elements, wl, dx)
    for _ in range(9):
        out = propagate_through_system_jax(E, elements, wl, dx)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    benchmark(lambda: propagate_through_system_jax(E, elements, wl, dx))
    benchmark.extra_info['first_call_ms'] = first_s * 1e3
    benchmark.extra_info['speedup_first_to_warm'] = (
        first_s / benchmark.stats.stats.mean
        if benchmark.stats.stats.mean > 0 else float('inf'))


# ===========================================================================
# gerchberg_saxton_jax
# ===========================================================================

def test_bench_gerchberg_saxton_jax_first_vs_warm(benchmark):
    """``gerchberg_saxton_jax`` 50-iter first call vs 10th call."""
    from lumenairy.analysis.phase_retrieval import _GS_KERNEL_CACHE
    N = 64
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    src = np.exp(-(X**2 + Y**2) / 0.5**2).astype(np.float32)
    tgt = np.exp(-(X**2 + Y**2) / 0.3**2).astype(np.float32)

    n_iter = 50
    _GS_KERNEL_CACHE.pop(n_iter, None)

    first_s = _time_one(lm.gerchberg_saxton_jax,
                         src, tgt, n_iter=n_iter, seed=0)
    for _ in range(9):
        _ = lm.gerchberg_saxton_jax(src, tgt, n_iter=n_iter, seed=0)
    benchmark(lambda: lm.gerchberg_saxton_jax(
        src, tgt, n_iter=n_iter, seed=0))
    benchmark.extra_info['first_call_ms'] = first_s * 1e3
    benchmark.extra_info['speedup_first_to_warm'] = (
        first_s / benchmark.stats.stats.mean
        if benchmark.stats.stats.mean > 0 else float('inf'))


# ===========================================================================
# error_reduction_jax
# ===========================================================================

def test_bench_error_reduction_jax_first_vs_warm(benchmark):
    """``error_reduction_jax`` 50-iter first call vs 10th call."""
    from lumenairy.analysis.phase_retrieval import _ER_KERNEL_CACHE
    N = 64
    meas = np.ones((N, N), dtype=np.float32)
    sup = np.ones((N, N), dtype=bool)
    n_iter = 50
    _ER_KERNEL_CACHE.pop(n_iter, None)

    first_s = _time_one(lm.error_reduction_jax,
                         meas, sup, n_iter=n_iter, seed=0)
    for _ in range(9):
        _ = lm.error_reduction_jax(meas, sup, n_iter=n_iter, seed=0)
    benchmark(lambda: lm.error_reduction_jax(
        meas, sup, n_iter=n_iter, seed=0))
    benchmark.extra_info['first_call_ms'] = first_s * 1e3
    benchmark.extra_info['speedup_first_to_warm'] = (
        first_s / benchmark.stats.stats.mean
        if benchmark.stats.stats.mean > 0 else float('inf'))


# ===========================================================================
# through_focus_scan_jax
# ===========================================================================

def test_bench_through_focus_scan_jax_first_vs_warm(benchmark):
    """``through_focus_scan_jax`` first call vs 10th call (the inner
    ASM kernel jit cache is the load-bearing piece)."""
    from lumenairy.analysis.through_focus import (
        _THROUGH_FOCUS_SCAN_JAX_CACHE,
    )
    N, dx, wl = 64, 5e-6, 1.55e-6
    sigma = 10e-6
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X**2 + Y**2) / (2*sigma**2)).astype(np.complex64)
    zs = np.linspace(-1e-3, 1e-3, 7)

    # v4.13.1 (L7): clear the jit cache before the first-call timing so
    # a re-run inside the same process actually measures the cold
    # (trace+compile) path, matching the four sibling benchmarks above.
    _THROUGH_FOCUS_SCAN_JAX_CACHE.clear()
    first_s = _time_one(
        lm.through_focus_scan_jax,
        E, dx=dx, wavelength=wl, z_values=zs,
        ideal_peak=float(np.max(np.abs(E)**2)),
        bandlimit=True)
    for _ in range(9):
        _ = lm.through_focus_scan_jax(
            E, dx=dx, wavelength=wl, z_values=zs,
            ideal_peak=float(np.max(np.abs(E)**2)),
            bandlimit=True)
    benchmark(lambda: lm.through_focus_scan_jax(
        E, dx=dx, wavelength=wl, z_values=zs,
        ideal_peak=float(np.max(np.abs(E)**2)),
        bandlimit=True))
    benchmark.extra_info['first_call_ms'] = first_s * 1e3
    benchmark.extra_info['speedup_first_to_warm'] = (
        first_s / benchmark.stats.stats.mean
        if benchmark.stats.stats.mean > 0 else float('inf'))
