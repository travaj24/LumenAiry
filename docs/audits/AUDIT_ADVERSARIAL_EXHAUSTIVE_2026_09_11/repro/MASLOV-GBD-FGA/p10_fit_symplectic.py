"""Probe 10: Maslov canonical-map fit -- conditioning, and the symplectic
identities dOPD/ds2 = n2*v2 - n1*v1.ds1/ds2 and dOPD/dv2 = -n1*v1.ds1/dv2."""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.elements.lenses import (_multi_indices_total_degree, _fit_normaliser)
from lumenairy._math.chebyshev import (chebyshev_vandermonde as CV,
                                       chebyshev_derivative_vandermonde as CVd)
warnings.simplefilter("ignore")

lam = 1.0e-6
presc = la.make_singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', aperture=1.5e-3)
surfaces = rt.surfaces_from_prescription(presc)
r_ap = 0.5*presc['aperture_diameter']
na = 0.05                      # deliberately non-collimated chart

def cheb_nodes(n):
    i = np.arange(n); return np.cos((i+0.5)*np.pi/n)
nf = np = None
import numpy as np
nfs, nps = 16, 16
hx = cheb_nodes(nfs); px = cheb_nodes(nps)
HX, HY, PX, PY = np.meshgrid(hx, hx, px, px, indexing='ij')
HX, HY, PX, PY = (a.ravel() for a in (HX,HY,PX,PY))
keep = (PX**2+PY**2) <= 1.0
HX,HY,PX,PY = HX[keep],HY[keep],PX[keep],PY[keep]
s1x, s1y = HX*r_ap, HY*r_ap
v1x, v1y = PX*na, PY*na
Ndir = np.sqrt(np.maximum(1-v1x**2-v1y**2, 0.0))
rays = rt.RayBundle(x=s1x.copy(), y=s1y.copy(), z=np.zeros_like(s1x),
                    L=v1x.copy(), M=v1y.copy(), N=Ndir, wavelength=lam,
                    alive=np.ones(len(s1x), bool), opd=np.zeros(len(s1x)))
tr = rt.trace(rays, surfaces, lam); ex = tr.image_rays; al = ex.alive
s2x, s2y = ex.x[al], ex.y[al]
v2x, v2y = ex.L[al], ex.M[al]
opd_w = (ex.opd[al]-rays.opd[al])/lam
S1x, S1y, V1x, V1y = s1x[al], s1y[al], v1x[al], v1y[al]
print(f"alive rays: {al.sum()}/{al.size}")

s2xc,s2xh = _fit_normaliser(s2x); s2yc,s2yh = _fit_normaliser(s2y)
v2xc,v2xh = _fit_normaliser(v2x); v2yc,v2yh = _fit_normaliser(v2y)
u1 = (s2x-s2xc)/s2xh; u2 = (s2y-s2yc)/s2yh
u3 = (v2x-v2xc)/v2xh; u4 = (v2y-v2yc)/v2yh

print("\n--- design-matrix conditioning (4-D total-degree Chebyshev) ---")
for order in (4,6,8,10):
    mi = _multi_indices_total_degree(4, order); M=len(mi)
    T1,T2,T3,T4 = CV(u1,order),CV(u2,order),CV(u3,order),CV(u4,order)
    A = np.empty((u1.size, M))
    for j,(k1,k2,k3,k4) in enumerate(mi): A[:,j] = T1[k1]*T2[k2]*T3[k3]*T4[k4]
    s = np.linalg.svd(A, compute_uv=False)
    G = A.T@A
    print(f"  order={order:2d}  M={M:4d}  n_rays={u1.size}  cond(A)={s[0]/s[-1]:.4e}"
          f"  cond(A^T A)={np.linalg.cond(G):.4e}  rays/M={u1.size/M:.2f}")

order = 6
mi = _multi_indices_total_degree(4, order); M=len(mi)
T1,T2,T3,T4 = CV(u1,order),CV(u2,order),CV(u3,order),CV(u4,order)
d3,d4 = CVd(u3,order), CVd(u4,order)
d1,d2 = CVd(u1,order), CVd(u2,order)
A = np.empty((u1.size, M))
for j,(k1,k2,k3,k4) in enumerate(mi): A[:,j] = T1[k1]*T2[k2]*T3[k3]*T4[k4]
RHS = np.column_stack([opd_w, S1x, S1y, V1x, V1y])
co,*_ = np.linalg.lstsq(A, RHS, rcond=None)
c_opd,c_s1x,c_s1y,c_v1x,c_v1y = (co[:,i] for i in range(5))
pred = A@co
print(f"\n--- fit residual RMS (order {order}) ---")
for nm,i,sc in (('opd[waves]',0,1),('s1x[um]',1,1e6),('s1y[um]',2,1e6),
                ('v1x',3,1),('v1y',4,1)):
    print(f"  {nm:11s}: {np.sqrt(np.mean((RHS[:,i]-pred[:,i])**2))*sc:.4e}")

def dA(which):
    B = np.empty((u1.size, M))
    for j,(k1,k2,k3,k4) in enumerate(mi):
        t = [T1[k1],T2[k2],T3[k3],T4[k4]]
        t[which] = [d1[k1],d2[k2],d3[k3],d4[k4]][which]
        B[:,j] = t[0]*t[1]*t[2]*t[3]
    return B
A_du1, A_du2, A_du3, A_du4 = dA(0),dA(1),dA(2),dA(3)

# physical derivatives
dopd_ds2x = (A_du1@c_opd)/s2xh; dopd_ds2y = (A_du2@c_opd)/s2yh   # waves/m
dopd_dv2x = (A_du3@c_opd)/v2xh; dopd_dv2y = (A_du4@c_opd)/v2yh   # waves
ds1x_dv2x = (A_du3@c_s1x)/v2xh; ds1x_dv2y = (A_du4@c_s1x)/v2yh
ds1y_dv2x = (A_du3@c_s1y)/v2xh; ds1y_dv2y = (A_du4@c_s1y)/v2yh
ds1x_ds2x = (A_du1@c_s1x)/s2xh; ds1x_ds2y = (A_du2@c_s1x)/s2yh
ds1y_ds2x = (A_du1@c_s1y)/s2xh; ds1y_ds2y = (A_du2@c_s1y)/s2yh
V1xf, V1yf = A@c_v1x, A@c_v1y

print("\n--- symplectic identity (1):  dOPD/dv2 = -n1 (v1 . ds1/dv2) / lambda ---")
for ax,(lhs,rx,ry) in (('x',(dopd_dv2x, ds1x_dv2x, ds1y_dv2x)),
                       ('y',(dopd_dv2y, ds1x_dv2y, ds1y_dv2y))):
    rhs = -(V1xf*rx + V1yf*ry)/lam
    print(f"   v2{ax}: RMS lhs={np.sqrt(np.mean(lhs**2)):.4e}  RMS(lhs-rhs)="
          f"{np.sqrt(np.mean((lhs-rhs)**2)):.4e}  rel={np.sqrt(np.mean((lhs-rhs)**2))/np.sqrt(np.mean(lhs**2)):.3e}")

print("\n--- symplectic identity (2):  dOPD/ds2 = [n2 v2 - n1 (v1 . ds1/ds2)]/lambda ---")
for ax,(lhs,v2c,rx,ry) in (('x',(dopd_ds2x, v2x, ds1x_ds2x, ds1y_ds2x)),
                           ('y',(dopd_ds2y, v2y, ds1x_ds2y, ds1y_ds2y))):
    rhs = (v2c - (V1xf*rx + V1yf*ry))/lam
    print(f"   s2{ax}: RMS lhs={np.sqrt(np.mean(lhs**2)):.4e}  RMS(lhs-rhs)="
          f"{np.sqrt(np.mean((lhs-rhs)**2)):.4e}  rel={np.sqrt(np.mean((lhs-rhs)**2))/np.sqrt(np.mean(lhs**2)):.3e}")

print("\n--- saddle of OPD alone: which input direction does it select? ---")
# grad_v2 OPD = 0  <=>  v1 . ds1/dv2 = 0  <=>  v1 = 0 (nonsingular ds1/dv2)
g = np.hypot(dopd_dv2x, dopd_dv2y)
sel = g < np.percentile(g, 2)
print(f"   at the 2% smallest |grad_v2 OPD| rays: |v1| mean={np.hypot(V1x,V1y)[sel].mean():.4e}"
      f"  (chart NA={na});  over ALL rays |v1| mean={np.hypot(V1x,V1y).mean():.4e}")
