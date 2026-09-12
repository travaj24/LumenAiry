from common import *
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d, rcwa_jones_1d
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
def iso(e): return e*np.eye(3, dtype=complex)
S=24
# --- (1) the CROSSED-cell raise ---
print("== fff_nv on a CROSSED cell must RAISE (CONVENTIONS section 11) ==")
cc = np.zeros((S,S,3,3),dtype=complex); cc[...]=iso(1.0); cc[6:18,6:18]=iso(12.25)
try:
    pmm_jones_2d(Px,Py,cc,nsub,nsup,dep,wl,degree=7,n_orders=3,formulation="fff_nv")
    print("  !! NO RAISE")
except Exception as e:
    print("  raised:", type(e).__name__, str(e)[:110])
print("  rcwa_jones_2d fff_nv on the same crossed cell:")
try:
    S2=48; cb=np.zeros((S2,S2,3,3),dtype=complex); cb[...]=iso(1.0); cb[12:36,12:36]=iso(12.25)
    o,R,T,J = rcwa_jones_2d(Px,Py,cb,nsub,nsup,dep,wl,n_orders_x=3,n_orders_y=3,formulation="fff_nv")
    print("    OK, Jxx=",J[0,0])
except Exception as e: print("    raised:", type(e).__name__, str(e)[:110])
# --- an ANISOTROPIC crossed cell? and a SEPARABLE-but-tensor cell ---
print()
print("== fff_nv SEPARABLE reduction vs Li-factorised RCWA-2D, high-contrast Si stripes ==")
cx = np.zeros((S,S,3,3),dtype=complex); cx[...]=iso(1.0); cx[6:18,:]=iso(12.25)
S2=96; cxb = np.zeros((S2,S2,3,3),dtype=complex); cxb[...]=iso(1.0); cxb[24:72,:]=iso(12.25)
# 1-D oracle
eps_r = iso(12.25); eps_g = iso(1.0)
o1,R1,T1,J1 = rcwa_jones_1d(Px, eps_r, eps_g, nsub, nsup, dep, 0.5, wl, angle=0.0, n_orders=80, formulation="li")
i0 = int(np.where(o1==0)[0][0])
print(f"  1-D RCWA oracle (n=80,li): Jr_tm(x)={J1[0,0]:.8f}  Jr_te(y)={J1[1,1]:.8f}")
print("  n |  PMM fff_nv Jxx          | PMM laurent Jxx          | PMM li Jxx               | RCWA li Jxx")
for nn in (3,5,7,9,11,13):
    row=[f"  {nn:2d}"]
    for form in ("fff_nv","laurent","li"):
        try:
            o,R,T,J = pmm_jones_2d(Px,Py,cx,nsub,nsup,dep,wl,degree=11,n_orders=nn,formulation=form)
            row.append(f"| {J[0,0]: .8f} ")
        except Exception as e: row.append(f"| ERR {type(e).__name__} ")
    try:
        o,R,T,J = rcwa_jones_2d(Px,Py,cxb,nsub,nsup,dep,wl,n_orders_x=nn,n_orders_y=nn,formulation="li")
        row.append(f"| {J[0,0]: .8f}")
    except Exception as e: row.append(f"| ERR {e}")
    print("".join(row), flush=True)
