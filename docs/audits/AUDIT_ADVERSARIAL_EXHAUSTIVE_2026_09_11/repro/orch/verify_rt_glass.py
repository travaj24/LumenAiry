import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
import lumenairy as la
from lumenairy.glass import get_glass_index
# 1) glass rows vs refractiveindex.info
try:
    import refractiveindex as ri
    db = ri.RefractiveIndexMaterial if hasattr(ri,'RefractiveIndexMaterial') else None
except Exception as e:
    db = None; print("refractiveindex import failed:", e)
print("1) bundled Sellmeier vs catalogue at 587.56 nm")
for g in ['N-BK7','N-BAF52','N-LAK33A','N-LAK33B','N-SF11']:
    nb = get_glass_index(g, 587.5618e-9)
    ncat = None
    if db is not None:
        try:
            m = db(shelf='specs', book='SCHOTT-optical', page=g); ncat = m.get_refractive_index(587.5618)
        except Exception as e:
            try:
                m = db(shelf='glass', book='SCHOTT-optical', page=g); ncat = m.get_refractive_index(587.5618)
            except Exception as e2:
                ncat = f"lookup failed ({type(e2).__name__})"
    print(f"   {g:9s} bundled n_d = {nb:.6f}   catalogue = {ncat if isinstance(ncat,str) else f'{ncat:.6f}'}")
# 2) conic false miss
print("2) conic intersection at h > |R| (R=10.84 mm, k=-0.6; conic valid to h=17.1 mm)")
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace
rx = {'surfaces':[{'radius':10.84e-3,'conic':-0.6,'glass_before':'AIR','glass_after':'N-BK7','semi_diameter':12e-3},
                  {'radius':np.inf,'conic':0.0,'glass_before':'N-BK7','glass_after':'AIR','semi_diameter':12e-3}],'thicknesses':[8e-3]}
hs = np.array([8e-3, 10.0e-3, 10.8e-3, 10.9e-3, 11.4e-3])
b = _make_bundle(x=hs, y=np.zeros_like(hs), L=np.zeros_like(hs), M=np.zeros_like(hs), wavelength=1e-6)
r = trace(b, surfaces_from_prescription(rx), 1e-6).image_rays
print("   h [mm]:", (hs*1e3).round(2).tolist()); print("   alive :", r.alive.tolist(), " error_code:", getattr(r,'error_code',None))
# 3) seidel ignores conic
print("3) seidel_coefficients S1 for a mirror R=-200 mm at k = 0 and k = -1 (parabola should be 0):")
from lumenairy.raytrace import seidel_coefficients
for k in (0.0, -1.0):
    rxm = {'surfaces':[{'radius':-200e-3,'conic':k,'glass_before':'AIR','glass_after':'MIRROR','is_mirror':True,'semi_diameter':25e-3}],
           'thicknesses':[], 'aperture_diameter':50e-3}
    try:
        s = seidel_coefficients(surfaces_from_prescription(rxm), 1e-6, aperture_radius=25e-3) if 'aperture_radius' in seidel_coefficients.__code__.co_varnames else seidel_coefficients(surfaces_from_prescription(rxm), 1e-6)
        S1 = getattr(s,'S1',None); S1 = S1 if S1 is not None else (s[0] if isinstance(s,(tuple,list)) else s)
        print(f"   k={k:+.1f}: S1 = {np.sum(S1) if hasattr(S1,'__len__') else S1}")
    except Exception as e:
        print(f"   k={k:+.1f}: call failed: {type(e).__name__}: {str(e)[:150]}")
# 4) turbulence screen variance vs lattice PSD sum (light version)
print("4) generate_turbulence_screen variance / sum(PSD*df^2):")
from lumenairy.elements.elements import generate_turbulence_screen
import inspect; print("   signature:", str(inspect.signature(generate_turbulence_screen))[:160])
