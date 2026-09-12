"""RW test 1: low-NA scalar Airy limit + absolute normalisation."""
import sys, warnings
import numpy as np
from scipy.special import j1, jn_zeros
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus

lam = 633e-9
NA  = 0.05
f   = 5e-3
Np  = 512
dxp = 2e-6
Nf  = 2048

xp = (np.arange(Np) - Np/2)*dxp
X, Y = np.meshgrid(xp, xp)
R = np.hypot(X, Y)
pupil = (R <= f*NA).astype(np.complex128)      # uniform, hard rim

with warnings.catch_warnings():
    warnings.simplefilter('error', RuntimeWarning)   # make any RW warning loud
    Ex, Ey, Ez, xf, yf = richards_wolf_focus(
        pupil, lam, NA, f, dxp, N_focal=Nf, polarization='x')

I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
k = 2*np.pi/lam
dxf = lam*f/(Nf*dxp)
print(f"dx_focal = {dxf*1e9:.3f} nm ; window = {Nf*dxf*1e6:.2f} um")
print(f"|Ex|max {np.abs(Ex).max():.6e}  |Ey|max {np.abs(Ey).max():.3e}  "
      f"|Ez|max {np.abs(Ez).max():.3e}")

# ---- absolute peak vs scalar Fraunhofer  E0 = pi a^2/(lam f) -------------
a = f*NA
pred_peak = np.pi*a*a/(lam*f)
meas_peak = np.abs(Ex).max()
print(f"peak |Ex| measured {meas_peak:.6e}  predicted pi a^2/(lam f) "
      f"{pred_peak:.6e}   ratio {meas_peak/pred_peak:.6f}")

# ---- radial profile vs [2 J1(v)/v]^2 ------------------------------------
Xf, Yf = np.meshgrid(xf, yf)
Rf = np.hypot(Xf, Yf)
v  = k*NA*Rf
with np.errstate(invalid='ignore', divide='ignore'):
    airy = np.where(v == 0, 1.0, (2*j1(v)/np.where(v == 0, 1, v))**2)
Ipk = I.max()
In  = I/Ipk

# restrict comparison to the first few Airy rings (r < 5 * first zero)
r1 = jn_zeros(1, 1)[0]/(k*NA)
sel = Rf <= 5*r1
num = np.sqrt(np.sum((In[sel]-airy[sel])**2))
den = np.sqrt(np.sum(airy[sel]**2))
print(f"first-zero radius (theory 0.6098 lam/NA) = {r1*1e6:.4f} um")
print(f"relative L2 error of normalised intensity vs Airy (r<5r1): {num/den:.4e}")
print(f"max abs deviation                                        : "
      f"{np.max(np.abs(In[sel]-airy[sel])):.4e}")

# ---- measured first zero from the azimuthal average ---------------------
nb = 800
rmax = 4*r1
bins = np.linspace(0, rmax, nb+1)
idx = np.digitize(Rf.ravel(), bins)-1
ok = (idx >= 0) & (idx < nb)
prof = np.bincount(idx[ok], weights=In.ravel()[ok], minlength=nb)
cnt  = np.bincount(idx[ok], minlength=nb)
prof = prof/np.maximum(cnt, 1)
rc = 0.5*(bins[:-1]+bins[1:])
good = cnt > 0
# first local minimum
p = prof[good]; rr = rc[good]
imin = None
for i in range(2, len(p)-2):
    if p[i] < p[i-1] and p[i] <= p[i+1] and p[i] < 0.05:
        imin = i; break
# parabolic refine
i = imin
den2 = (p[i-1]-2*p[i]+p[i+1])
shift = 0.5*(p[i-1]-p[i+1])/den2 if den2 != 0 else 0.0
r0 = rr[i] + shift*(rr[i+1]-rr[i])
print(f"MEASURED first-zero radius = {r0*1e6:.4f} um   "
      f"theory {r1*1e6:.4f} um   rel err {abs(r0-r1)/r1:.4e}")
print(f"MEASURED first-zero as multiple of lam/NA = {r0/(lam/NA):.5f}  "
      f"(theory 0.60976)")
