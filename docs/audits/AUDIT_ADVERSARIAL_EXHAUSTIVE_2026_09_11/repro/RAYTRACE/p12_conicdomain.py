"""RAYTRACE probe: (i) conic OUT-OF-DOMAIN policy numpy(NaN) vs jax(0.0);
(ii) the h > |R| false miss, cross-backend."""
import os, sys
os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace._conic_core import conic_sag
from lumenairy.elements.lenses import surface_sag_general
from lumenairy.raytrace import trace, Surface, surfaces_from_prescription
from lumenairy.raytrace.trace import _make_bundle

WL = 1.31e-6
print('=' * 78)
print('(i) sag OUT OF THE CONIC DOMAIN: the two backends disagree')
print('=' * 78)
R, k = 20e-3, 2.0           # oblate: valid only for h < R/sqrt(1+k)
hlim = R / np.sqrt(1 + k)
print(f'   R={R*1e3} mm  k={k}  valid h < {hlim*1e3:.4f} mm')
for h in (5e-3, 11.0e-3, 11.6e-3, 15e-3):
    s_np = surface_sag_general(np.array([h**2]), R, k, None)[0]
    s_jx = conic_sag(np.array([h]), np.array([0.0]), R, k, (), xp=np)
    print(f'   h={h*1e3:6.2f} mm : elements.surface_sag_general (NumPy trace) '
          f'= {s_np: .6e} ;  _conic_core.conic_sag (JAX/ADRT) = {s_jx[0]: .6e}')
print('   -> out of domain the NumPy raytrace sees NaN (-> RAY_MISSED) while')
print('      the shared core used by the JAX kernels returns a FINITE 0.0 --')
print('      a phantom flat surface at the vertex plane that Newton can')
print('      converge onto.')

print()
print('=' * 78)
print('(ii) h > |R| conic false-miss: NumPy vs JAX vs the analytic ADRT')
print('=' * 78)
presc = {
    'surfaces': [
        {'radius': 10.84e-3, 'conic': -0.6, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 11.5e-3},
        {'radius': np.inf, 'glass_before': 'N-BK7', 'glass_after': 'air',
         'semi_diameter': 11.5e-3},
        {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
         'semi_diameter': np.inf},
    ],
    'thicknesses': [8e-3, 10e-3, 0.0],
}
ys = np.array([5e-3, 9e-3, 10.5e-3, 10.9e-3, 11.4e-3])
surfs = surfaces_from_prescription(presc)
rb = _make_bundle(np.zeros_like(ys), ys, np.zeros_like(ys),
                  np.zeros_like(ys), WL)
res = trace(rb, surfs, WL, output_filter='last').image_rays
print(f'   NumPy trace   alive = {res.alive}   err = {res.error_code}')
try:
    import jax.numpy as jnp
    from lumenairy.raytrace.jax_trace import trace_jax, make_jax_ray_state
    st = make_jax_ray_state(x=jnp.zeros(len(ys)), y=jnp.asarray(ys),
                            z=jnp.zeros(len(ys)), L=jnp.zeros(len(ys)),
                            M=jnp.zeros(len(ys)), N=jnp.ones(len(ys)))
    rj = trace_jax(st, presc, WL)
    print(f'   trace_jax     alive = {np.asarray(rj.alive)}')
except Exception as e:
    print('   trace_jax failed:', type(e).__name__, e)
from lumenairy.raytrace.differential import ray_transfer_jacobian_analytic
dt = ray_transfer_jacobian_analytic(
    np.zeros_like(ys), ys.copy(), np.zeros_like(ys), np.zeros_like(ys),
    surfs, WL)
print(f'   analytic ADRT alive = {dt.alive}  (exact conic quadratic --')
print('                  it has NO false miss)')
