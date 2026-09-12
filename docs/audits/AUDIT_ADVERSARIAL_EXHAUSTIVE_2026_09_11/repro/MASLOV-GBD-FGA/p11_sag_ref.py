"""Probe 11: is the Maslov exit chart referenced to the last-surface VERTEX PLANE
or to the curved surface intersection?"""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.elements.lenses import _multi_indices_total_degree, _fit_normaliser
from lumenairy._math.chebyshev import (chebyshev_vandermonde as CV,
                                       chebyshev_derivative_vandermonde as CVd)
warnings.simplefilter("ignore")
lam = 1.0e-6
presc = la.make_singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', aperture=1.5e-3)
surfaces = rt.surfaces_from_prescription(presc)
r_ap, na = 0.5*presc['aperture_diameter'], 0.05
def cn(n): i=np.arange(n); return np.cos((i+0.5)*np.pi/n)
h, p = cn(16), cn(16)
HX,HY,PX,PY = (a.ravel() for a in np.meshgrid(h,h,p,p, indexing='ij'))
kp = (PX**2+PY**2)<=1.0
HX,HY,PX,PY = HX[kp],HY[kp],PX[kp],PY[kp]
s1x,s1y,v1x,v1y = HX*r_ap, HY*r_ap, PX*na, PY*na
Nd = np.sqrt(np.maximum(1-v1x**2-v1y**2,0))
rays = rt.RayBundle(x=s1x.copy(),y=s1y.copy(),z=np.zeros_like(s1x),L=v1x.copy(),
                    M=v1y.copy(),N=Nd,wavelength=lam,alive=np.ones(len(s1x),bool),
                    opd=np.zeros(len(s1x)))
ex = rt.trace(rays, surfaces, lam).image_rays; al = ex.alive
print(f"exit z (should be 0 at a vertex-plane reference): "
      f"min={ex.z[al].min():.4e}  max={ex.z[al].max():.4e}  RMS={np.sqrt(np.mean(ex.z[al]**2)):.4e} m"
      f"   -> {np.sqrt(np.mean(ex.z[al]**2))/lam:.1f} waves of path")
# what the PLANE-referenced chart would be:
zc = ex.z[al]; Nz = ex.N[al]
t = -zc/Nz
s2x_p = ex.x[al] + t*ex.L[al]; s2y_p = ex.y[al] + t*ex.M[al]
opd_p = (ex.opd[al] + t) / lam                 # exit medium is air (n=1)
opd_c = ex.opd[al]/lam
print(f"vertex-plane correction: transverse shift RMS = {np.sqrt(np.mean((s2x_p-ex.x[al])**2))*1e6:.4f} um,"
      f"  OPD shift RMS = {np.sqrt(np.mean((opd_p-opd_c)**2)):.3f} waves, PV = {np.ptp(opd_p-opd_c):.3f} waves")

# re-test symplectic identity (2) with the CURVED-surface correction term
s2x, s2y = ex.x[al], ex.y[al]; v2x, v2y = ex.L[al], ex.M[al]
V1x, V1y = v1x[al], v1y[al]; S1x, S1y = s1x[al], s1y[al]
opd_w = ex.opd[al]/lam
c1,h1=_fit_normaliser(s2x); c2,h2=_fit_normaliser(s2y)
c3,h3=_fit_normaliser(v2x); c4,h4=_fit_normaliser(v2y)
u1,u2,u3,u4 = (s2x-c1)/h1,(s2y-c2)/h2,(v2x-c3)/h3,(v2y-c4)/h4
order=6; mi=_multi_indices_total_degree(4,order); M=len(mi)
T=[CV(u,order) for u in (u1,u2,u3,u4)]; D=[CVd(u,order) for u in (u1,u2,u3,u4)]
A=np.empty((u1.size,M))
for j,k in enumerate(mi): A[:,j]=T[0][k[0]]*T[1][k[1]]*T[2][k[2]]*T[3][k[3]]
def dA(w):
    B=np.empty((u1.size,M))
    for j,k in enumerate(mi):
        t_=[T[0][k[0]],T[1][k[1]],T[2][k[2]],T[3][k[3]]]
        t_[w]=D[w][k[w]]
        B[:,j]=t_[0]*t_[1]*t_[2]*t_[3]
    return B
co,*_=np.linalg.lstsq(A, np.column_stack([opd_w,S1x,S1y,V1x,V1y]), rcond=None)
c_opd,c_s1x,c_s1y,c_v1x,c_v1y = (co[:,i] for i in range(5))
A1,A2 = dA(0),dA(1)
dopd_ds2x=(A1@c_opd)/h1; ds1x_ds2x=(A1@c_s1x)/h1; ds1y_ds2x=(A1@c_s1y)/h1
V1xf,V1yf = A@c_v1x, A@c_v1y
# plane reference:  dOPD/ds2x = [v2x - v1.ds1/ds2x]/lam
rhs_plane = (v2x - (V1xf*ds1x_ds2x + V1yf*ds1y_ds2x))/lam
# curved reference z = sigma(s2): dS/ds2 = n2*(v2 + N2*dsigma/ds2)
R = -6.0e-3
sig_p = (s2x/R)/np.sqrt(np.maximum(1-(s2x**2+s2y**2)/R**2, 1e-12))   # dsigma/ds2x
rhs_curved = (v2x + ex.N[al]*sig_p - (V1xf*ds1x_ds2x + V1yf*ds1y_ds2x))/lam
for nm, rhs in (('PLANE-referenced identity', rhs_plane),
                ('CURVED-surface identity  ', rhs_curved)):
    r = np.sqrt(np.mean((dopd_ds2x-rhs)**2))/np.sqrt(np.mean(dopd_ds2x**2))
    print(f"  {nm}: relative residual = {r:.4e}")
