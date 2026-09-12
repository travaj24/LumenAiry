import os, sys, time, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np, lumenairy as la
from lumenairy.optimize.driver import design_optimize, _fd_grad_pure
from lumenairy.optimize import DesignParameterization, FocalLengthMerit, MinThicknessMerit
from lumenairy.optimize.jax_merits import JaxMeritTerm

print("np.asarray(None, float64) =", repr(np.asarray(None, dtype=np.float64)),
      " ndim =", np.asarray(None, dtype=np.float64).ndim)

# ---- JIT vs NumPy parity for the MultiFieldMerit kernel ----
from lumenairy.optimize import _merit_jit as MJ
rng = np.random.default_rng(0)
for N in (64, 256):
    kX = rng.normal(size=(N, N)); kY = rng.normal(size=(N, N))
    mask = rng.random((N, N)) > 0.3
    a = MJ._multi_field_tilt_phasor_masked(0.3, -0.2, kX, kY, mask, np.complex128)
    tilt = 0.3*kX + (-0.2)*kY
    b = np.where(mask, np.exp(1j*tilt), 0.0).astype(np.complex128)
    print(f"  JIT parity N={N}: max|d| = {np.abs(a-b).max():.3e}  (numba used: {N*N >= MJ._MULTI_FIELD_JIT_MIN_PIXELS})")

# ---- merit-evaluation counting: 3-variable singlet ----
tmpl = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
param = DesignParameterization(
    template=tmpl,
    free_vars=[('surfaces', 0, 'radius'), ('surfaces', 1, 'radius'), ('thicknesses', 0)],
    bounds=[(20e-3, 200e-3), (-200e-3, -20e-3), (1e-3, 10e-3)])
merits = [FocalLengthMerit(target=100e-3, weight=1.0), MinThicknessMerit(min_thickness=2e-3)]
calls = [0]
_orig = FocalLengthMerit.evaluate
def counted(self, ctx):
    calls[0] += 1
    return _orig(self, ctx)
FocalLengthMerit.evaluate = counted
t0=time.perf_counter()
res = design_optimize(param, merits, wavelength=587.6e-9, N=64, dx=20e-6,
                      method='L-BFGS-B', max_iter=5, verbose=False)
t1=time.perf_counter()
FocalLengthMerit.evaluate = _orig
print(f"ray-only 3-var L-BFGS-B, max_iter=5: {calls[0]} merit evals, "
      f"{res.iterations} iters, {t1-t0:.2f}s, merit={res.merit:.4g}, efl={res.context_final.efl*1e3:.3f} mm")
print("   -> evals/iteration =", calls[0]/max(res.iterations,1), " (n+1 = 4 expected for scipy 2-point FD)")

# ---- wave-leg cost: how many propagations per merit eval? ----
import lumenairy.optimize.core as _core
prop_calls = [0]; scan_calls = [0]
_op = _core.apply_real_lens if hasattr(_core,'apply_real_lens') else None
from lumenairy.elements import lenses as _L
_orig_arl = _L.apply_real_lens
_orig_tfs = _core.through_focus_scan
def c_arl(*a, **k):
    prop_calls[0]+=1; return _orig_arl(*a, **k)
def c_tfs(E, dx, wl, zs, **k):
    scan_calls[0]+=len(zs); return _orig_tfs(E, dx, wl, zs, **k)
import lumenairy.optimize.driver as _drv
_drv.apply_real_lens = c_arl
_core.through_focus_scan = c_tfs
from lumenairy.optimize import StrehlMerit
merits2 = [FocalLengthMerit(target=100e-3), StrehlMerit(min_strehl=0.8)]
calls2 = [0]
def counted2(self, ctx):
    calls2[0]+=1; return _orig(self, ctx)
FocalLengthMerit.evaluate = counted2
t0=time.perf_counter()
try:
    res2 = design_optimize(param, merits2, wavelength=587.6e-9, N=128, dx=20e-6,
                           method='L-BFGS-B', max_iter=2, verbose=False)
    t1=time.perf_counter()
    print(f"wave-leg 3-var, max_iter=2, N=128: {calls2[0]} merit evals, "
          f"{prop_calls[0]} apply_real_lens calls, {scan_calls[0]} through-focus slices, {t1-t0:.2f}s")
    print(f"   -> per merit eval: {prop_calls[0]/max(calls2[0],1):.1f} lens props + "
          f"{scan_calls[0]/max(calls2[0],1):.1f} focus-scan propagations")
except Exception as e:
    print("wave-leg run failed:", type(e).__name__, str(e)[:200])
finally:
    FocalLengthMerit.evaluate = _orig
    _drv.apply_real_lens = _orig_arl
    _core.through_focus_scan = _orig_tfs

# ---- JAX grad vs finite differences ----
try:
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    def build_args(x):
        return (x[0], x[1], x[2])
    def fn(R1, R2, d):
        n = 1.5168
        # thick-lens power (lensmaker), a smooth analytic function of the 3 vars
        P = (n-1)*(1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))
        return (1.0/P - 0.1)**2
    jm = JaxMeritTerm(fn, weight=1.0, build_args=build_args, real_part=True, needs_ray=False)
    x = np.array([50e-3, -50e-3, 4e-3])
    g_jax = jm.gradient_at_x(x)
    f = lambda xv: float(np.asarray(fn(*build_args(xv))))
    g_fd = _fd_grad_pure(f, x, eps=1e-6, scale_floor=np.array([1e-6,1e-6,1e-6]))
    rel = np.abs(g_jax-g_fd)/np.maximum(np.abs(g_fd),1e-30)
    print("JAX grad :", g_jax)
    print("FD  grad :", g_fd)
    print("rel err  :", rel, " max =", rel.max())
except Exception as e:
    print("JAX probe failed:", type(e).__name__, str(e)[:200])
