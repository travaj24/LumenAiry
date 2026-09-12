"""RAYTRACE probes 9 + 10: differential Jacobian vs central FD of trace();
JAX parity with the NumPy trace (x64 on/off), jax.grad of the exit OPL.
"""
import os, sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
X64 = os.environ.get('LUM_X64', '1') == '1'
if X64:
    os.environ['JAX_ENABLE_X64'] = '1'
from lumenairy.raytrace import (trace, Surface, make_ray)
from lumenairy.raytrace.differential import (ray_transfer_jacobian,
                                              ray_transfer_jacobian_analytic)
from lumenairy.raytrace.trace import _make_bundle

WL = 1.31e-6
np.set_printoptions(precision=6, suppress=True, linewidth=150)


def build(with_cb=False):
    s = [Surface(radius=50e-3, conic=-0.3, semi_diameter=np.inf,
                 glass_before='air', glass_after='N-BK7', thickness=6e-3),
         Surface(radius=-80e-3, conic=0.2, semi_diameter=np.inf,
                 glass_before='N-BK7', glass_after='air', thickness=20e-3),
         Surface(radius=120e-3, semi_diameter=np.inf, glass_before='air',
                 glass_after='N-SF2', thickness=4e-3),
         Surface(radius=np.inf, semi_diameter=np.inf, glass_before='N-SF2',
                 glass_after='air', thickness=30e-3),
         Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                 glass_after='air', thickness=0.0)]
    if with_cb:
        s.insert(2, Surface(is_coordbrk=True, tilt_x_deg=3.0, tilt_y_deg=-2.0,
                            decenter_x_m=0.3e-3, decenter_y_m=-0.2e-3,
                            glass_before='air', glass_after='air',
                            thickness=5e-3))
        s[1].thickness = 15e-3
    return s


def fd_jacobian_via_trace(x0, y0, ux0, uy0, surfaces, h=(2e-7, 2e-7, 2e-6, 2e-6)):
    """Independent central-difference Jacobian built by calling trace() directly."""
    def state(xx, yy, uu, vv):
        inv = 1.0 / np.sqrt(1 + uu * uu + vv * vv)
        rb = _make_bundle(np.atleast_1d(xx), np.atleast_1d(yy),
                          np.atleast_1d(uu * inv), np.atleast_1d(vv * inv), WL)
        r = trace(rb, surfaces, WL, output_filter='last').image_rays
        return np.array([r.x[0], r.y[0], r.L[0] / r.N[0], r.M[0] / r.N[0]])
    base = np.array([x0, y0, ux0, uy0])
    J = np.zeros((4, 4))
    for d in range(4):
        p = base.copy(); m = base.copy()
        p[d] += h[d]; m[d] -= h[d]
        J[:, d] = (state(*p) - state(*m)) / (2 * h[d])
    return J


print('=' * 76)
print('9. Analytic ADRT Jacobian vs central FD of trace()')
print('=' * 76)
for with_cb in (False, True):
    surfs = build(with_cb)
    pts = [(0.0, 0.0, 0.0, 0.0), (2e-3, -1e-3, 0.01, -0.02),
           (5e-3, 4e-3, -0.03, 0.02)]
    for (x0, y0, ux0, uy0) in pts:
        try:
            dt = ray_transfer_jacobian_analytic(
                np.array([x0]), np.array([y0]), np.array([ux0]),
                np.array([uy0]), surfs, WL)
            Ja = dt.jacobian[0]
        except NotImplementedError as e:
            print('   analytic NotImplemented:', e); continue
        Jf = fd_jacobian_via_trace(x0, y0, ux0, uy0, surfs)
        dfd = ray_transfer_jacobian(np.array([x0]), np.array([y0]),
                                     np.array([ux0]), np.array([uy0]),
                                     surfs, WL)
        Jd = dfd.jacobian[0]
        rel = np.abs(Ja - Jf) / np.maximum(np.abs(Jf), 1e-9)
        rel2 = np.abs(Jd - Jf) / np.maximum(np.abs(Jf), 1e-9)
        print(f'  cb={with_cb} ray=({x0:.0e},{y0:.0e},{ux0:+.2f},{uy0:+.2f}): '
              f'max rel |J_analytic - J_FDtrace| = {rel.max():.3e} ; '
              f'max rel |J_libFD - J_FDtrace| = {rel2.max():.3e}')
        print(f'      opd: analytic={dt.opd[0]:.12e}  libFD={dfd.opd[0]:.12e}  '
              f'd={abs(dt.opd[0]-dfd.opd[0]):.2e}')

print()
print('=' * 76)
print('10. JAX parity with the NumPy trace  (x64 = %s)' % X64)
print('=' * 76)
import jax
import jax.numpy as jnp
print('  jax x64 enabled:', jax.config.read('jax_enable_x64'))
from lumenairy.raytrace.jax_trace import (trace_jax, make_jax_ray_state,
                                           trace_jax_with_params)
presc = {
    'surfaces': [
        {'radius': 50e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -80e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'},
        {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air'},
    ],
    'thicknesses': [6e-3, 100e-3, 0.0],
    'aperture_diameter': 40e-3,
}
from lumenairy.raytrace import surfaces_from_prescription
surfs = surfaces_from_prescription(presc)
ys = np.linspace(-12e-3, 12e-3, 9)
rb = _make_bundle(np.zeros_like(ys), ys, np.zeros_like(ys),
                  np.zeros_like(ys), WL)
res = trace(rb, surfs, WL, output_filter='last').image_rays
st = make_jax_ray_state(x=jnp.zeros_like(jnp.asarray(ys)),
                        y=jnp.asarray(ys), z=jnp.zeros_like(jnp.asarray(ys)),
                        L=jnp.zeros_like(jnp.asarray(ys)),
                        M=jnp.zeros_like(jnp.asarray(ys)),
                        N=jnp.ones_like(jnp.asarray(ys)))
rj = trace_jax(st, presc, WL)
print('  dtype of jax opd:', np.asarray(rj.opd).dtype)
print(f'  max |dy|   numpy vs jax = {np.abs(np.asarray(rj.y)-res.y).max():.3e} m')
print(f'  max |dopd| numpy vs jax = {np.abs(np.asarray(rj.opd)-res.opd).max():.3e} m')
print(f'  max |dL|   numpy vs jax = {np.abs(np.asarray(rj.L)-res.L).max():.3e}')

print()
print('  jax.grad of the mean exit OPL w.r.t. R1 vs finite differences:')


def opl_of_R1(R1):
    s = make_jax_ray_state(x=jnp.zeros(9), y=jnp.asarray(ys), z=jnp.zeros(9),
                           L=jnp.zeros(9), M=jnp.zeros(9), N=jnp.ones(9))
    radii = jnp.stack([R1, jnp.float64(-80e-3), jnp.float64(jnp.inf)])
    r = trace_jax_with_params(s, presc, WL, radii=radii)
    return jnp.mean(r.opd)


g = jax.grad(opl_of_R1)(jnp.float64(50e-3))
h = 1e-8
fd = (opl_of_R1(jnp.float64(50e-3 + h)) - opl_of_R1(jnp.float64(50e-3 - h))) / (2 * h)
print(f'    jax.grad = {float(g):.10e}   FD = {float(fd):.10e}   '
      f'rel = {abs(float(g)-float(fd))/max(abs(float(fd)),1e-30):.3e}')

print()
print('  recompile-per-call check (trace_jax timing, 200 calls):')
import time
_ = trace_jax(st, presc, WL)
t0 = time.perf_counter()
for _ in range(200):
    r = trace_jax(st, presc, WL)
    r.x.block_until_ready()
t1 = time.perf_counter()
print(f'    {1e6*(t1-t0)/200:.1f} us / call')
t0 = time.perf_counter()
for _ in range(200):
    res = trace(rb, surfs, WL, output_filter='last')
t1 = time.perf_counter()
print(f'    numpy trace (9 rays): {1e6*(t1-t0)/200:.1f} us / call')
