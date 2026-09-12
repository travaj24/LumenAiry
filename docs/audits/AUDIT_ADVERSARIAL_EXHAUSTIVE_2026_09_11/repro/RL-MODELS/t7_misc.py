import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.elements._lens_real as LR
from lumenairy.elements._lens_real import (apply_real_lens,
    _get_displaced_cos_grid, set_pointwise_cos_grid_cache_budget,
    _build_displaced_cos_luts, _DISPLACED_LUT_CACHE)

lam = 0.55e-6

print("="*74)
print("B. tangent_facet_remap fold guard: is it scored over the ILLUMINATED pupil")
print("   or over the whole grid (incl. dark corners)?")
print("="*74)
for N, dx in [(512, 4.0e-6), (512, 12.0e-6), (512, 20.0e-6), (512, 30.0e-6)]:
    ap = 2.0e-3
    axis = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(axis, axis)
    E = np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
    rx = dict(surfaces=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
                        dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')],
              thicknesses=[2.5e-3], aperture_diameter=ap)
    corner = 0.5*np.hypot(N*dx, N*dx)
    try:
        R = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                            surface_model='tangent_facet_remap')
        print(f"  N={N} dx={dx*1e6:5.1f}um window={N*dx*1e3:6.2f}mm "
              f"corner_r={corner*1e3:5.2f}mm ap={ap*1e3:.1f}mm  -> OK "
              f"P={float(np.sum(np.abs(R)**2)):.6g}")
    except ValueError as e:
        print(f"  N={N} dx={dx*1e6:5.1f}um window={N*dx*1e3:6.2f}mm "
              f"corner_r={corner*1e3:5.2f}mm ap={ap*1e3:.1f}mm  -> REFUSED: "
              f"{str(e)[:150]}")

print()
print("="*74)
print("C. pointwise cos-grid cache: STALE HIT on a mutated sag_callable closure")
print("="*74)
class Freeform:
    def __init__(self, a): self.a = a
    def __call__(self, xs, ys): return self.a*(xs**2 + ys**2)
ff = Freeform(5.0)
s0 = dict(radius=np.inf, glass_before='AIR', glass_after='N-BK7',
          sag_callable=ff, decenter=(1e-9, 0.0))
s1 = dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR',
          decenter=(1e-9, 0.0))
surfaces = [s0, s1]; thick = [2.5e-3]
set_pointwise_cos_grid_cache_budget(64)
g1 = _get_displaced_cos_grid(surfaces, thick, lam, 1.0e-3, 128, 128,
                             4e-6, 4e-6, None, None)
ff.a = -5000.0          # same object, completely different surface
g2 = _get_displaced_cos_grid(surfaces, thick, lam, 1.0e-3, 128, 128,
                             4e-6, 4e-6, None, None)
LR.clear_pointwise_cos_grid_cache()
g3 = _get_displaced_cos_grid(surfaces, thick, lam, 1.0e-3, 128, 128,
                             4e-6, 4e-6, None, None)
print(f"  cached hit identical to pre-mutation trace : {np.array_equal(g1[0][0], g2[0][0])}")
print(f"  cold rebuild differs from cached           : "
      f"{not np.allclose(g2[0][0], g3[0][0])}  "
      f"max|dcos| = {float(np.max(np.abs(g2[0][0]-g3[0][0]))):.4e}")
set_pointwise_cos_grid_cache_budget(0)

print()
print("="*74)
print("D. _build_displaced_ray_map: exit leg hardcoded n=1 even when the last")
print("   surface's glass_after is NOT air")
print("="*74)
from lumenairy.elements._lens_real import _build_displaced_ray_map
sA = [dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
      dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')]
sB = [dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
      dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='N-BK7')]  # immersed exit
hA, hoA, oA = _build_displaced_ray_map(sA, [2.5e-3], lam, 1.0e-3)
hB, hoB, oB = _build_displaced_ray_map(sB, [2.5e-3], lam, 1.0e-3)
# for sB the last surface does not refract (n1==n2) -> ray is straight; the
# vertex-plane referencing leg should use n=1.5168, not 1.0
print(f"  n_after=AIR   OPL span over pupil = {float(oA.max()-oA.min())*1e6:8.3f} um")
print(f"  n_after=N-BK7 OPL span over pupil = {float(oB.max()-oB.min())*1e6:8.3f} um")
print("  (the exit referencing leg is `opl += 1.0*t_f` -- line ~1552)")

print()
print("="*74)
print("E. meridional LUT crossing-height monotonicity with a converging conjugate")
print("="*74)
for conj in (None, -0.030, -0.0205, 0.05):
    slope = None if conj is None else (lambda h, s=conj: np.asarray(h)/s)
    luts = _build_displaced_cos_luts(sA, [2.5e-3], lam, 1.0e-3,
                                     carrier_slope=slope)
    for i, (h, ci, co) in enumerate(luts):
        d = np.diff(h)
        nd = int(np.sum(d <= 0))
        print(f"  conjugate={conj}: surface {i}: n={h.size} non-increasing steps="
              f"{nd} h range=[{h.min()*1e6:.2f},{h.max()*1e6:.2f}]um "
              f"dup/reversal={'YES' if nd else 'no'}")
