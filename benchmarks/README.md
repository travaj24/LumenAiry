# LumenAiry benchmarks

Per-feature pytest-benchmark scripts captured alongside the v4.12.x
performance work.  Each file follows the naming convention
``test_bench_<area>.py`` and lives alongside the unit tests' module
structure where possible.

## Running

```
# All benchmarks
python -m pytest benchmarks/ --benchmark-only -v

# One area, save baseline
python -m pytest benchmarks/test_bench_fft.py --benchmark-save=v4_11_2

# Compare against a saved baseline
python -m pytest benchmarks/test_bench_fft.py --benchmark-compare=v4_11_2
```

## Conventions

- Each benchmark file pairs with a correctness-pinning test under
  ``tests/unit/test_perf_v4_12_<id>.py`` so we never trade correctness
  for speed.
- Benchmarks default to a small, fast workload (so the suite runs in
  reasonable time) but each file documents the "real-world" workload
  it was designed for.
- Median-of-N timing; ``--benchmark-min-rounds=5`` per file.
- When a benchmark depends on an optional dep (pyfftw, jax, cupy,
  numba), skip with ``pytest.importorskip(...)`` rather than failing.

## Per-release baselines saved

- ``v4_11_2`` -- pre-v4.12 work, captured 2026-05-16.

## Workload conventions

The audit identified four dominant workload patterns; each benchmark
file labels which workload(s) it targets:

- **OPT** -- ``design_optimize`` / FD Jacobians
- **FWD** -- forward ``apply_real_lens`` through prescriptions
- **TRACE** -- ray-trace + Seidel sweeps
- **WIDE** -- wide-field PSF (asymptotic / HF / RW / MC tolerancing)
