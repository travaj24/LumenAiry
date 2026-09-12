"""RAYTRACE: JAX vs NumPy on an OBLATE conic beyond its valid radius but
inside |R| (so the sphere discriminant is still positive), plus a check
that the trace_jax eager jit cache is actually hit."""
import os, sys, time
os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import trace, surfaces_from_prescription
from lumenairy.raytrace.trace import _make_bundle
import jax.numpy as jnp
from lumenairy.raytrace import jax_trace as JT

WL = 1.31e-6
R, k = 20e-3, 2.0
hlim = R / np.sqrt(1 + k)
presc = {
    'surfaces': [
        {'radius': R, 'conic': k, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': np.inf, 'glass_before': 'N-BK7', 'glass_after': 'air'},
    ],
    'thicknesses': [10e-3, 0.0],
}
ys = np.array([5e-3, 11.0e-3, 12.0e-3, 15.0e-3, 18.0e-3])
print(f'R={R*1e3} mm  k={k}  conic valid to h={hlim*1e3:.3f} mm ; |R|={R*1e3} mm')
surfs = surfaces_from_prescription(presc)
rb = _make_bundle(np.zeros_like(ys), ys, np.zeros_like(ys),
                  np.zeros_like(ys), WL)
r = trace(rb, surfs, WL, output_filter='last').image_rays
print(f'  NumPy : alive={r.alive}  err={r.error_code}')
st = JT.make_jax_ray_state(x=jnp.zeros(len(ys)), y=jnp.asarray(ys),
                           z=jnp.zeros(len(ys)), L=jnp.zeros(len(ys)),
                           M=jnp.zeros(len(ys)), N=jnp.ones(len(ys)))
rj = JT.trace_jax(st, presc, WL)
print(f'  JAX   : alive={np.asarray(rj.alive)}')
print(f'  JAX y  = {np.asarray(rj.y)}')
print(f'  JAX L  = {np.asarray(rj.L)}')
print(f'  JAX opd= {np.asarray(rj.opd)}')

print()
print('trace_jax eager jit-cache behaviour:')
JT.clear_trace_jax_cache()
t0 = time.perf_counter(); JT.trace_jax(st, presc, WL); t1 = time.perf_counter()
print(f'  1st call (compile): {1e3*(t1-t0):.2f} ms ; cache size = '
      f'{len(JT._TRACE_JAX_CACHE)}')
t0 = time.perf_counter()
for _ in range(50):
    o = JT.trace_jax(st, presc, WL)
    o.x.block_until_ready()
t1 = time.perf_counter()
print(f'  warm calls: {1e6*(t1-t0)/50:.1f} us/call ; cache size = '
      f'{len(JT._TRACE_JAX_CACHE)}')
