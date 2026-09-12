import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
wl=587.6e-9; N=256; dx=25e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r2=X**2+Y**2
E0=np.exp(-r2/(1.2e-3)**2).astype(complex)
base={'surfaces':[{'radius':100e-3,'conic':0.0,'glass_before':'AIR','glass_after':'N-BK7'},
                  {'radius':-100e-3,'conic':0.0,'glass_before':'N-BK7','glass_after':'AIR'}],'thicknesses':[3e-3],'aperture_diameter':4e-3}
print("1) fast_analytic_phase=True:")
try:
    apply_real_lens_traced(E0, prescription=base, wavelength=wl, dx=dx, ray_subsample=4, fast_analytic_phase=True); print("   OK")
except Exception as e: print("   RAISED", type(e).__name__, ":", str(e)[:90])
print("2) form_error suppression (250 nm PV astigmatic figure error on S1):")
fe = 125e-9*(X**2-Y**2)/(2e-3)**2
withfe={**base,'surfaces':[dict(base['surfaces'][0], form_error=fe), dict(base['surfaces'][1])]}
m = r2 <= (1.5e-3)**2
for name, fn, kw in (('analytic', apply_real_lens, {}), ('traced', apply_real_lens_traced, {'ray_subsample':4})):
    a=fn(E0, prescription=base, wavelength=wl, dx=dx, **kw); b=fn(E0, prescription=withfe, wavelength=wl, dx=dx, **kw)
    d=np.angle(b[m]*np.conj(a[m])); print(f"   {name:9s}: max|dphi| with vs without form_error = {np.max(np.abs(d)):.3e} rad  (expected ~ k0*(n-1)*250nm*... = {2*np.pi/wl*0.5168*125e-9:.3e} rad at the edge)")
