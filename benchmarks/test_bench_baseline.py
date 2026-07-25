"""Baseline benchmarks captured pre-v4.12.

One representative case per workload pattern.  These exist to anchor
``--benchmark-compare`` runs as we work through the v4.12.x perf work.
Cheap enough that the whole file finishes in <60s on a typical laptop.

Workloads pinned here:

- FFT      -- single ASM propagate, 1024x1024 complex128 (FFT-bound)
- FWD      -- apply_real_lens through a 4-surface BK7 singlet, 512x512
- TRACE    -- 1k-ray trace through a 4-surface prescription
- SEIDEL   -- single seidel_coefficients call (paraxial walk)
- ZERNIKE  -- zernike_basis_matrix build, 21 modes, 256x256
- THROUGH  -- through_focus_scan NumPy, 7-point scan
- WIDE     -- richards_wolf_focus, single z-plane, 256x256 pupil
"""
from __future__ import annotations

import numpy as np

import lumenairy as lm

# ---------------------------------------------------------------------
# FFT-bound: single ASM propagate
# ---------------------------------------------------------------------

def test_bench_asm_propagate_1024(benchmark):
    N, dx, wl = 1024, 5e-6, 1.55e-6
    z = 1e-3
    E = np.ones((N, N), dtype=np.complex128)
    # Warm the H-cache so we benchmark the steady-state call.
    _ = lm.angular_spectrum_propagate(E, z, wl, dx)
    benchmark(lm.angular_spectrum_propagate, E, z, wl, dx)


# ---------------------------------------------------------------------
# FWD: apply_real_lens through a 4-surface BK7 singlet
# ---------------------------------------------------------------------

def test_bench_apply_real_lens_512(benchmark):
    N, dx, wl = 512, 8e-6, 1.55e-6
    presc = lm.make_doublet(
        R1=50e-3, R2=-30e-3, R3=-50e-3,
        d1=4e-3, d2=2e-3,
        glass1='N-BK7', glass2='N-SF11',
        aperture=20e-3)
    E = np.ones((N, N), dtype=np.complex128)
    benchmark(lm.apply_real_lens, E, prescription=presc, wavelength=wl, dx=dx)


# ---------------------------------------------------------------------
# TRACE: 1k-ray trace through a 4-surface prescription
# ---------------------------------------------------------------------

def test_bench_trace_1k_rays(benchmark):
    from lumenairy.raytrace import make_fan, surfaces_from_prescription, trace
    presc = lm.make_doublet(
        R1=50e-3, R2=-30e-3, R3=-50e-3,
        d1=4e-3, d2=2e-3,
        glass1='N-BK7', glass2='N-SF11',
        aperture=20e-3)
    surfs = surfaces_from_prescription(presc)
    wl = 0.5876e-6
    fan = make_fan('y', 10e-3, n_rays=1001, field_angle=0.0,
                   wavelength=wl)
    benchmark(trace, fan, surfs, wl)


# ---------------------------------------------------------------------
# SEIDEL: paraxial walk + Seidel sums
# ---------------------------------------------------------------------

def test_bench_seidel_doublet(benchmark):
    from lumenairy.raytrace import seidel_coefficients, surfaces_from_prescription
    presc = lm.make_doublet(
        R1=50e-3, R2=-30e-3, R3=-50e-3,
        d1=4e-3, d2=2e-3,
        glass1='N-BK7', glass2='N-SF11',
        aperture=20e-3)
    surfs = surfaces_from_prescription(presc)
    wl = 0.5876e-6
    benchmark(seidel_coefficients, surfs, wl,
              field_angle=np.radians(1.0))


# ---------------------------------------------------------------------
# ZERNIKE: 21-mode basis matrix on a 256x256 grid
# ---------------------------------------------------------------------

def test_bench_zernike_basis_21x_256(benchmark):
    from lumenairy.analysis.core import zernike_basis_matrix
    N = 256
    x = (np.arange(N) - N / 2) * 1.0
    y = (np.arange(N) - N / 2) * 1.0
    X, Y = np.meshgrid(x, y)
    pupil_radius = float(N / 2)
    benchmark(zernike_basis_matrix, 21, X, Y, pupil_radius)


# ---------------------------------------------------------------------
# THROUGH: 7-point through-focus scan, NumPy backend
# ---------------------------------------------------------------------

def test_bench_through_focus_7pt(benchmark):
    from lumenairy.analysis.through_focus import through_focus_scan
    N, dx, wl = 256, 8e-6, 1.55e-6
    z_arr = np.linspace(95e-3, 105e-3, 7)
    presc = lm.make_singlet(
        R1=51.5e-3, R2=float('inf'),
        d=2e-3, glass='N-BK7', aperture=5e-3)
    E = np.ones((N, N), dtype=np.complex128)
    E_exit = lm.apply_real_lens(
        E, prescription=presc, wavelength=wl, dx=dx)
    benchmark(through_focus_scan, E_exit, dx, wl, z_arr, verbose=False)


# ---------------------------------------------------------------------
# WIDE: Richards-Wolf focus at a single z-plane, 256x256 pupil
# ---------------------------------------------------------------------

def test_bench_richards_wolf_256(benchmark):
    Np, wl, NA, f = 256, 1.55e-6, 0.5, 1e-3
    dx_pupil = 1.5 * (f * NA) / (Np / 2)
    pupil = np.ones((Np, Np), dtype=np.complex128)
    benchmark(lm.richards_wolf_focus,
              pupil, wavelength=wl, NA=NA, f=f,
              dx_pupil=dx_pupil, N_focal=Np,
              polarization='x')
