"""Through-focus scan benchmarks for v4.12 FFT-hoist optimisation.

Workload: **OPT** (``design_optimize`` / FD Jacobians) and the
inner-loop hot path for ``MultiFieldMerit`` / ``MultiWavelengthMerit``
and Monte-Carlo tolerancing -- all of which call
``through_focus_scan`` once per outer-iter.

v4.11.2 baseline: each z-plane re-ran the full input FFT and rebuilt
the H transfer function from scratch (with H-cache hit on the second
run + within the same call after the first z).

v4.12.0 hoist: pre-computes the input FFT, kx/ky, and the propagating
mask above the z-loop; only ``exp(1j * kz * z) * bandlimit_z`` is
built per-z.  Expected speedup ~2-3x on N>=256.

Workloads pinned here:

- 7-pt N=256 ASM    -- matches the v4_11_2 baseline anchor exactly.
- 31-pt N=256 ASM   -- typical ``design_optimize`` workload.
- 31-pt N=512 ASM   -- heavier workload exercising larger FFT.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.analysis.through_focus import through_focus_scan


def _build_singlet_exit_field(N=256, dx=8e-6, wl=1.55e-6):
    """Build the same exit-pupil field the v4_11_2 baseline uses so
    7-pt N=256 timing is directly comparable across releases."""
    presc = lm.make_singlet(
        R1=51.5e-3, R2=float('inf'),
        d=2e-3, glass='N-BK7', aperture=5e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_exit = lm.apply_real_lens(
        E_in, prescription=presc, wavelength=wl, dx=dx)
    return E_exit


# ---------------------------------------------------------------------
# 7-pt N=256: same workload as v4_11_2 baseline test_bench_through_focus_7pt
# ---------------------------------------------------------------------

def test_bench_through_focus_7pt_n256(benchmark):
    N, dx, wl = 256, 8e-6, 1.55e-6
    z_arr = np.linspace(95e-3, 105e-3, 7)
    E_exit = _build_singlet_exit_field(N=N, dx=dx, wl=wl)
    benchmark(through_focus_scan, E_exit, dx, wl, z_arr, verbose=False)


# ---------------------------------------------------------------------
# 31-pt N=256: representative design_optimize / MultiFieldMerit workload
# ---------------------------------------------------------------------

def test_bench_through_focus_31pt_n256(benchmark):
    N, dx, wl = 256, 8e-6, 1.55e-6
    z_arr = np.linspace(95e-3, 105e-3, 31)
    E_exit = _build_singlet_exit_field(N=N, dx=dx, wl=wl)
    benchmark(through_focus_scan, E_exit, dx, wl, z_arr, verbose=False)


# ---------------------------------------------------------------------
# 31-pt N=512: heavier workload (exercises the larger FFT path)
# ---------------------------------------------------------------------

def test_bench_through_focus_31pt_n512(benchmark):
    N, dx, wl = 512, 4e-6, 1.55e-6
    z_arr = np.linspace(95e-3, 105e-3, 31)
    E_exit = _build_singlet_exit_field(N=N, dx=dx, wl=wl)
    benchmark(through_focus_scan, E_exit, dx, wl, z_arr, verbose=False)
