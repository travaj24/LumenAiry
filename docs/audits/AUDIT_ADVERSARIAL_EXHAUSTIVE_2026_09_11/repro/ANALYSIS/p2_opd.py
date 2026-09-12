"""ANALYSIS probe 2: opd.py -- unwrap validity, reference sphere, sampling rule."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.opd import (wave_opd_1d, wave_opd_2d, check_opd_sampling,
                                    depth_of_focus, opd_pv_rms, remove_wavefront_modes)

lam = 633e-9
k0 = 2*np.pi/lam

def make_converging(N, dx, f, ap, extra=None):
    x = (np.arange(N) - N/2)*dx
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    phase = -k0*R2/(2*f)
    if extra is not None:
        phase = phase + extra(X, Y)
    amp = (R2 <= (ap/2)**2).astype(float)
    return amp*np.exp(1j*phase), X, Y

print("=== A. sign convention: OPD of a converging wavefront ===")
N, dx, f, ap = 512, 2e-6, 0.05, 400e-6
E, X, Y = make_converging(N, dx, f, ap)
samp = check_opd_sampling(dx, lam, ap, f, verbose=False)
print(f"  check_opd_sampling: dx_max={samp['dx_max']*1e6:.4f} um, margin={samp['margin']:.3f}, "
      f"phase/sample={samp['phase_per_sample']:.4f} rad")
print(f"  hand: dphi_edge = k*(ap/2)/f*dx = {k0*(ap/2)/f*dx:.4f} rad")
co, op = wave_opd_1d(E, dx, lam, axis='x', aperture=ap)
r = co
exact = -r**2/(2*f)
# remove piston
res = (op - op[len(op)//2]) - (exact - exact[len(exact)//2])
print(f"  1-D OPD sign: opd(edge)={op[0]:.4e} m, analytic -r^2/2f={exact[0]:.4e} m")
print(f"  1-D residual vs analytic (piston removed): max={np.abs(res).max():.3e} m "
      f"= {np.abs(res).max()/lam:.3e} waves")

print()
print("=== B. wave_opd_2d on a 20-wave defocus sphere, various sampling ===")
# choose f so the edge OPD = 20 waves: r_max^2/(2f) = 20*lam
for frac in (4.0, 2.0, 1.0, 0.5):   # dx = dx_max/frac ; frac<1 => undersampled
    ap2 = 400e-6
    f2 = (ap2/2)**2/(2*20*lam)
    dxm = lam*f2/ap2
    dx2 = dxm/frac
    N2 = int(np.ceil(ap2/dx2/2))*2 + 64
    E2, X2, Y2 = make_converging(N2, dx2, f2, ap2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xo, Yo, opd = wave_opd_2d(E2, dx2, lam, aperture=ap2)
    exact2 = -(X2**2+Y2**2)/(2*f2)
    m = np.isfinite(opd)
    d = (opd - exact2)[m]
    d = d - np.median(d)
    print(f"  dx=dx_max/{frac:4.1f} N={N2:4d} : max|err|={np.abs(d).max()/lam:10.4f} waves, "
          f"rms={np.sqrt((d**2).mean())/lam:9.4f} waves  frac_bad={np.mean(np.abs(d)>0.4*lam):.3f}")

print()
print("=== C. same, WITH f_ref (should be exact at any sampling) ===")
for frac in (2.0, 1.0, 0.5, 0.25):
    ap2 = 400e-6
    f2 = (ap2/2)**2/(2*20*lam)
    dxm = lam*f2/ap2
    dx2 = dxm/frac
    N2 = int(np.ceil(ap2/dx2/2))*2 + 64
    E2, X2, Y2 = make_converging(N2, dx2, f2, ap2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xo, Yo, opd = wave_opd_2d(E2, dx2, lam, aperture=ap2, f_ref=f2)
    exact2 = -(X2**2+Y2**2)/(2*f2)
    m = np.isfinite(opd)
    d = (opd - exact2)[m]; d = d - np.median(d)
    print(f"  dx=dx_max/{frac:5.2f} N={N2:4d} : max|err|={np.abs(d).max()/lam:10.4f} waves "
          f" rms={np.sqrt((d**2).mean())/lam:9.4f} waves")

print()
print("=== D. 2-D unwrap on an ABERRATED (non-separable) wavefront, well sampled ===")
# 3 waves of trefoil on top of mild defocus. Nyquist-safe.
ap3, f3 = 400e-6, 2.0
N3, dx3 = 512, 1e-6
def trefoil(X, Y):
    rho = np.sqrt(X**2+Y**2)/(ap3/2); th = np.arctan2(Y, X)
    return k0*3*lam*np.sqrt(8)*(rho**3)*np.cos(3*th)
E3, X3, Y3 = make_converging(N3, dx3, f3, ap3, extra=trefoil)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _, _, opd3 = wave_opd_2d(E3, dx3, lam, aperture=ap3)
exact3 = -(X3**2+Y3**2)/(2*f3) + trefoil(X3, Y3)/k0
m3 = np.isfinite(opd3)
d3 = (opd3-exact3)[m3]; d3 = d3-np.median(d3)
print(f"  max|err| = {np.abs(d3).max()/lam:.4e} waves, rms={np.sqrt((d3**2).mean())/lam:.4e} waves")
print(f"  fraction of pixels off by >0.4 wave: {np.mean(np.abs(d3)>0.4*lam):.4f}")

print()
print("=== E. ANNULAR pupil (central obscuration) 2-D unwrap ===")
N5, dx5, f5, ap5 = 512, 1e-6, 2.0, 400e-6
E5, X5, Y5 = make_converging(N5, dx5, f5, ap5)
R5 = np.sqrt(X5**2+Y5**2)
E5 = E5*(R5 > 0.3*ap5/2)      # 30% obscuration
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _, _, opd5 = wave_opd_2d(E5, dx5, lam, aperture=ap5)
exact5 = -(X5**2+Y5**2)/(2*f5)
m5 = np.isfinite(opd5)
d5 = (opd5-exact5)[m5]; d5 = d5-np.median(d5)
print(f"  max|err| = {np.abs(d5).max()/lam:.4f} waves   frac>0.4wave = {np.mean(np.abs(d5)>0.4*lam):.4f}")

print()
print("=== F. depth_of_focus sanity ===")
print("  f/2, 550nm:", depth_of_focus(550e-9, 2.0), " expect 4.4e-6")
NA = 1/(2*2.0)
print("  lambda/(2 NA^2) =", 550e-9/(2*NA**2))

print()
print("=== G. check_opd_sampling: is the factor right for the 1-D unwrap? ===")
# empirical: sweep dx and find where wave_opd_1d first breaks
ap7, f7 = 400e-6, 0.02
dxm = lam*f7/ap7
print(f"  predicted dx_max = {dxm*1e6:.4f} um")
for mult in (0.4, 0.6, 0.8, 0.9, 0.98, 1.02, 1.2, 1.5):
    dx7 = dxm*mult
    N7 = int(np.ceil(ap7/dx7/2))*2 + 32
    E7, X7, Y7 = make_converging(N7, dx7, f7, ap7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        co7, op7 = wave_opd_1d(E7, dx7, lam, axis='x', aperture=ap7)
    ex7 = -co7**2/(2*f7)
    dd = (op7-ex7); dd -= dd[len(dd)//2]
    print(f"   dx = {mult:5.2f}*dx_max : max|err| = {np.abs(dd).max()/lam:10.4f} waves")
