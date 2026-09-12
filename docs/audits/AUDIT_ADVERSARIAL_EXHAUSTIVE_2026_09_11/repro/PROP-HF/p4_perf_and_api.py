"""PROP-HF p4: cost model / memory of the HF OPL quadrature, the hf free-space
return-type contract, and the dispatcher's view of hf / hfpi."""
import sys
import time
import tracemalloc
import warnings
import inspect
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel, propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_with_opl_callable as hf_opl)

LAM = 633e-9


def sec(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


sec("(1) hf OPL quadrature: cost per output pixel, and the O(N^4) ceiling")
for N in (64, 128, 256):
    dx = 1e-6
    E = np.ones((N, N), dtype=np.complex128)
    z = 1e-3

    def opl(a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / LAM

    xo = np.array([0.0, 1e-6, 2e-6, 3e-6])
    t0 = time.perf_counter()
    hf_opl(E, opl_fn=opl, output_grid_x=xo, output_grid_y=np.array([0.0]),
           input_grid_dx=dx, apply_van_vleck=True)
    dt = (time.perf_counter() - t0) / 4.0
    full = dt * N * N
    print(f"  N={N:4d}: {dt*1e3:8.3f} ms per OUTPUT PIXEL  =>  "
          f"{full:9.1f} s for a full {N}x{N} output   "
          f"({full/3600:.2f} h)")

sec("(2) tracemalloc peak of ONE output pixel at N=256, vs one field (8N^2 B)")
N, dx, z = 256, 1e-6, 1e-3
E = np.ones((N, N), dtype=np.complex128)


def opl(a1, b1, a2, b2):
    return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / LAM


for vv in (False, True):
    tracemalloc.start()
    hf_opl(E, opl_fn=opl, output_grid_x=np.array([0.0]),
           output_grid_y=np.array([0.0]), input_grid_dx=dx,
           apply_van_vleck=vv)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  apply_van_vleck={vv!s:5s}  peak = {peak/1e6:7.3f} MB  = "
          f"{peak/(8.0*N*N):6.2f} x (8 N^2 = one float64 grid = "
          f"{8.0*N*N/1e6:.3f} MB)")
print("  (16 opl_fn calls per output pixel for the Van Vleck stencil + 1 for Phi;")
print("   each opl_fn call allocates several full N^2 float64 temporaries)")

sec("(3) hf free-space return-type contract")
E = np.ones((32, 32), dtype=np.complex128)
r1 = propagate_huygens_fresnel(E, 1e-3, LAM, 2e-6)
print(f"  no output kwargs           -> {type(r1).__name__} "
      f"{getattr(r1, 'shape', None)}")
r2 = propagate_huygens_fresnel(E, 1e-3, LAM, 2e-6, output_dx=2e-6)
print(f"  output_dx == dx (no-op)    -> {type(r2).__name__} len={len(r2)} "
      f"elem0={type(r2[0]).__name__}")
r3 = propagate_huygens_fresnel(E, 1e-3, LAM, 2e-6, output_shape=(32, 32))
print(f"  output_shape == in shape   -> {type(r3).__name__} len={len(r3)}")
r4 = propagate_huygens_fresnel(E, 1e-3, LAM, 2e-6, output_dx=4e-6, output_shape=(16, 16))
print(f"  genuine resample           -> {type(r4).__name__} len={len(r4)} "
      f"field {r4[0].shape} dx {r4[1]:.3e}")
try:
    propagate_huygens_fresnel(E, 1e-3, LAM, 2e-6, output_shape=(16, 32))
except ValueError as e:
    print(f"  non-square output_shape    -> ValueError ({str(e)[:60]}...)")

sec("(4) does the dispatcher reach hf / hfpi, and with what contract?")
from lumenairy.propagators import dispatch  # noqa: E402
src = inspect.getsource(dispatch)
for tag in ("'hf'", "'hfpi'", '"hf"', '"hfpi"'):
    n = src.count(tag)
    print(f"  dispatch.py mentions {tag:8s}: {n} times")
try:
    from lumenairy.propagators.dispatch import _METHOD_REGISTRY  # noqa
    print("  registry:", sorted(_METHOD_REGISTRY))
except Exception as e:
    print(f"  (_METHOD_REGISTRY not importable: {type(e).__name__})")
# what does propagate(method='hf') return?
from lumenairy.propagators.dispatch import propagate  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    out = propagate(E, 1e-3, LAM, 2e-6, method='hf', return_result=False)
print(f"  propagate(method='hf') -> {type(out).__name__} "
      f"{getattr(out, 'shape', len(out) if hasattr(out, '__len__') else '')}")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    out2 = propagate(E, 1e-3, LAM, 2e-6, method='hf',
                     output_grid={'N': 16, 'dx': 4e-6}, return_result=False)
print(f"  propagate(method='hf', output_grid=...) -> {type(out2).__name__} "
      f"{getattr(out2, 'shape', len(out2) if hasattr(out2, '__len__') else '')}")
