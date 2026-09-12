"""PROP-HF p8: (a) float32 OPD through _hfpi_segment_trace; (b) clean HF
timing; (c) complex64 path in hf_opl actually runs in complex128."""
import sys
import time
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la  # noqa: E402
from lumenairy.propagators.hfpi import (  # noqa: E402
    init_paths_from_field, _hfpi_segment_trace)
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel_with_opl_callable as hf_opl)

LAM = 633e-9
print("=" * 70)
print("(a) complex64 source -> float32 opl -> RayBundle.opd through trace()")
print("=" * 70)
from lumenairy.raytrace import surfaces_from_prescription  # noqa: E402
presc = la.make_singlet(R1=50e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
                        aperture=10e-3)
surfs = surfaces_from_prescription(presc)
for dt_in in (np.complex128, np.complex64):
    E = np.ones((8, 8), dtype=dt_in)
    p = init_paths_from_field(E, 20e-6, n_paths=64, wavelength=LAM, rng=1,
                              cone_half_angle=0.02)
    print(f"  E_in {np.dtype(dt_in).name:10s}: paths.opl dtype = {p.opl.dtype}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = _hfpi_segment_trace(p, surfs, LAM)
    print(f"      after _hfpi_segment_trace: opl dtype = {q.opl.dtype}, "
          f"weights {q.weights.dtype}")
    ph = np.angle(np.asarray(q.weights)[np.asarray(q.alive)][:4])
    print(f"      first 4 alive weight phases: {np.round(ph, 8)}")

print()
print("=" * 70)
print("(b) HF OPL quadrature cost per output pixel (median of 5, warm)")
print("=" * 70)
for N in (128, 256, 512):
    dx, z = 1e-6, 1e-3
    E = np.ones((N, N), dtype=np.complex128)

    def opl(a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / LAM

    hf_opl(E, opl_fn=opl, output_grid_x=np.array([0.0]),
           output_grid_y=np.array([0.0]), input_grid_dx=dx)   # warm
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        hf_opl(E, opl_fn=opl, output_grid_x=np.array([0.0]),
               output_grid_y=np.array([0.0]), input_grid_dx=dx)
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f"  N_in={N:4d}: {med*1e3:8.3f} ms / output pixel  -> full {N}x{N} "
          f"output = {med*N*N/60:9.1f} min")

print()
print("=" * 70)
print("(c) complex64 E_in: does the quadrature stay in complex64?")
print("=" * 70)
N, dx, z = 128, 1e-6, 1e-3
for dt_in in (np.complex128, np.complex64):
    E = np.ones((N, N), dtype=dt_in)

    def opl(a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / LAM

    out = hf_opl(E, opl_fn=opl, output_grid_x=np.array([0.0]),
                 output_grid_y=np.array([0.0]), input_grid_dx=dx)
    # measure the intermediate dtype the way the source builds it
    S = np.zeros((N, N))
    kern = np.exp(2j * np.pi * S).astype(out.dtype)
    integ = (E * np.float64(1.0) * kern)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        hf_opl(E, opl_fn=opl, output_grid_x=np.array([0.0]),
               output_grid_y=np.array([0.0]), input_grid_dx=dx)
        ts.append(time.perf_counter() - t0)
    print(f"  E_in {np.dtype(dt_in).name:10s} -> out {out.dtype}, "
          f"integrand dtype {integ.dtype}, "
          f"{np.median(ts)*1e3:7.2f} ms/px")
