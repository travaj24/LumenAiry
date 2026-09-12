"""Probe 9: stale hit in _DISPLACED_LUT_CACHE (ON by default, no opt-in) when
a user-supplied GLASS_REGISTRY entry is re-pointed under the same NAME --
the key stores str(glass_name) only."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.glass import GLASS_REGISTRY, get_glass_index
import lumenairy.elements._lens_real as LR
from lumenairy.elements._lens_real import apply_real_lens, _get_displaced_cos_luts

lam = 0.55e-6
GLASS_REGISTRY['MYGLASS'] = lambda wl: 1.50
print("n(MYGLASS) =", get_glass_index('MYGLASS', lam))
sA = [dict(radius=+19.6e-3, glass_before='AIR',     glass_after='MYGLASS'),
      dict(radius=-27.4e-3, glass_before='MYGLASS', glass_after='AIR')]
l1 = _get_displaced_cos_luts(sA, [2.5e-3], lam, 1.0e-3, None, None)
GLASS_REGISTRY['MYGLASS'] = lambda wl: 1.90     # re-point, SAME name
try:
    from lumenairy.glass import clear_glass_cache
    clear_glass_cache()
except Exception:
    for nm in ('_GLASS_VALUE_CACHE', '_glass_cache', '_GLASS_CACHE'):
        obj = getattr(__import__('lumenairy.glass', fromlist=['x']), nm, None)
        if hasattr(obj, 'clear'):
            obj.clear()
    try:
        get_glass_index.cache_clear()
    except Exception: pass
print("n(MYGLASS) after re-point =", get_glass_index('MYGLASS', lam))
l2 = _get_displaced_cos_luts(sA, [2.5e-3], lam, 1.0e-3, None, None)   # cache HIT?
LR.clear_displaced_lut_cache()
l3 = _get_displaced_cos_luts(sA, [2.5e-3], lam, 1.0e-3, None, None)   # cold
print("  cached (l2) == pre-repoint (l1):",
      np.array_equal(l1[1][1], l2[1][1]))
print("  cold (l3) differs from cached (l2):",
      not np.allclose(l2[1][1], l3[1][1]),
      " max|dcos_in| =", float(np.max(np.abs(l2[1][1]-l3[1][1]))))
# effect through the public API
N=256; dx=8e-6; ap=1.6e-3
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
E=np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
rx=dict(surfaces=sA, thicknesses=[2.5e-3], aperture_diameter=ap)
GLASS_REGISTRY['MYGLASS'] = lambda wl: 1.50
LR.clear_displaced_lut_cache()
E1=apply_real_lens(E.copy(),prescription=rx,wavelength=lam,dx=dx,surface_model='displaced')
GLASS_REGISTRY['MYGLASS'] = lambda wl: 1.90
try:
    from lumenairy.glass import clear_glass_cache; clear_glass_cache()
except Exception: pass
E2=apply_real_lens(E.copy(),prescription=rx,wavelength=lam,dx=dx,surface_model='displaced')
LR.clear_displaced_lut_cache()
E3=apply_real_lens(E.copy(),prescription=rx,wavelength=lam,dx=dx,surface_model='displaced')
m=(X**2+Y**2)<=(0.8*ap/2)**2
print("  public API:  |E2-E3|max/|E3|max (stale LUT vs correct) =",
      f"{float(np.max(np.abs(E2-E3)[m])/np.max(np.abs(E3)[m])):.4e}")
