import sys, numpy as np, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"); sys.path.insert(0,".")
np.set_printoptions(precision=8, linewidth=240)
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S=24
def iso(e): return e*np.eye(3, dtype=complex)

# --- (b) 1-D grating: x-periodic vs y-periodic (90 deg rotation) ---
print("=== (b) 90-deg rotation covariance (1-D grating) ===")
cx = np.zeros((S,S,3,3),dtype=complex); cx[...] = iso(1.0)
cx[6:18,:,:,:] = iso(12.25)          # ridge occupying x in [0.25,0.75]P, uniform in y
cy = np.transpose(cx, (1,0,2,3)).copy()   # same pattern, y-periodic
for nn in (3,5):
  for thd,phid in ((0.0,0.0),(20.0,0.0)):
    th,ph = np.deg2rad(thd), np.deg2rad(phid)
    o,R,T,Jx = pmm_jones_2d(Px,Py,cx,nsub,nsup,dep,wl,theta=th,phi=ph,degree=7,n_orders=nn)
    # rotate the problem: for a y-periodic cell, incidence at phi+90 should map
    o2,R2,T2,Jy = pmm_jones_2d(Px,Py,cy,nsub,nsup,dep,wl,theta=th,phi=ph+np.pi/2,degree=7,n_orders=nn)
    # x<->y swap operator
    Sw = np.array([[0,1],[1,0]])
    Jy_sw = Sw@Jy@Sw
    print(f" n_orders={nn} th={thd}: max|Jx - swap(Jy)| = {np.max(np.abs(Jx-Jy_sw)):.3e}   |Jx|max={np.max(np.abs(Jx)):.4f}")
    if nn==3 and thd==0.0:
        print("   Jx=",Jx.ravel()); print("   Jy_sw=",Jy_sw.ravel())

# --- (c) mirror-symmetric cell, normal incidence -> zero cross-pol ---
print("=== (c) mirror-symmetric cell, normal incidence: cross-pol ===")
cm = np.zeros((S,S,3,3),dtype=complex); cm[...] = iso(1.0)
cm[6:18,4:20,:,:] = iso(12.25)      # rectangle, mirror-symmetric about both axes
o,R,T,J = pmm_jones_2d(Px,Py,cm,nsub,nsup,dep,wl,theta=0.0,phi=0.0,degree=7,n_orders=5)
print("  J=",J.ravel()); print("  |Jxy|=",abs(J[0,1]),"|Jyx|=",abs(J[1,0]))
o,R,T,Jn = pmm_jones_2d(Px,Py,cm,nsub,nsup,dep,wl,theta=0.0,phi=0.0,degree=7,n_orders=5,symmetry=False)
print("  symmetry=False: |Jxy|=",abs(Jn[0,1]),"|Jyx|=",abs(Jn[1,0]), " dJ(fold vs full)=",np.max(np.abs(J-Jn)))

# --- (d) C4 square pillar -> Jxx == Jyy at normal incidence ---
print("=== (d) C4 square pillar: Jxx vs Jyy ===")
c4 = np.zeros((S,S,3,3),dtype=complex); c4[...] = iso(1.0)
c4[6:18,6:18,:,:] = iso(12.25)
for form in ("laurent","li"):
  o,R,T,J = pmm_jones_2d(Px,Py,c4,nsub,nsup,dep,wl,theta=0.0,phi=0.0,degree=7,n_orders=5,formulation=form)
  print(f"  form={form}: Jxx={J[0,0]:.10f} Jyy={J[1,1]:.10f} |Jxx-Jyy|={abs(J[0,0]-J[1,1]):.3e} |Jxy|={abs(J[0,1]):.2e}")
  print(f"     R rows equal? {np.max(np.abs(R[0]-R[1])):.3e}  (should be 0 by C4... only after x<->y order swap)")
