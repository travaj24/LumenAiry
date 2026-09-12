"""Probe 9 (parallel_amp bitwise) + Probe 11 (complex64 dtype)."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
WL = common.WL; k0 = 2*np.pi/WL
N = 512; AP = 8e-3; dx = 1.3*AP/N
rx = common.plano_convex(R=60e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
w = 2e-3
E64 = (np.exp(-(X**2+Y**2)/w**2)).astype(np.complex128)

kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ea = apply_real_lens_traced(E64, parallel_amp=True, **kw)
    Eb = apply_real_lens_traced(E64, parallel_amp=False, **kw)
    Ec = apply_real_lens_traced(E64, parallel_amp=True, **kw)
print("=== Probe 9: parallel_amp ===")
print("  parallel==sequential bitwise :", np.array_equal(Ea, Eb),
      " maxabs diff =", np.abs(Ea-Eb).max())
print("  parallel run 1 == run 2      :", np.array_equal(Ea, Ec))
print("  relative L2 (par vs seq)     :",
      np.linalg.norm(Ea-Eb)/max(np.linalg.norm(Eb), 1e-300))

print("=== Probe 11: complex64 ===")
E32 = E64.astype(np.complex64)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    E32o = apply_real_lens_traced(E32, **kw)
    E64o = apply_real_lens_traced(E64, **kw)
print("  out dtypes:", E32o.dtype, E64o.dtype)
m = np.abs(E64o) > 1e-4*np.abs(E64o).max()
dph = np.angle(E32o[m].astype(np.complex128)*np.conj(E64o[m]))
print(f"  phase(c64) - phase(c128): rms={dph.std():.4e} rad "
      f"({dph.std()/k0*1e9:.4f} nm)  max={np.abs(dph).max():.4e} rad "
      f"({np.abs(dph).max()/k0*1e9:.4f} nm)")
da = (np.abs(E32o[m]).astype(np.float64)-np.abs(E64o[m]))/np.abs(E64o).max()
print(f"  |E| rel diff: rms={da.std():.3e} max={np.abs(da).max():.3e}")
# where would float32 enter the OPL?
print("  NOTE: k0*OPL_absolute at the vertex ~ "
      f"{k0*(1.5168*4e-3):.3e} rad; float32 ulp there = "
      f"{np.spacing(np.float32(k0*1.5168*4e-3)):.3e} rad")
