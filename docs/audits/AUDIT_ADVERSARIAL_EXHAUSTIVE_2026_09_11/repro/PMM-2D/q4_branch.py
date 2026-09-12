from common import *
from lumenairy.elements.rcwa._core import _sqrt_decay, _CUT_BAND_REL
print("== constructed on-cut cases for _sqrt_decay ==")
cases = {
 "exact -4 (Im=+0)":  np.array([-4.0+0.0j]),
 "exact -4 (Im=-0)":  np.array([complex(-4.0,-0.0)]),
 "-4 + 1e-18j":       np.array([-4.0+1e-18j]),
 "-4 - 1e-18j":       np.array([-4.0-1e-18j]),
 "+4 - 1e-18j (evan)":np.array([ 4.0-1e-18j]),
 "tiny +1e-20-1e-30j":np.array([1e-20-1e-30j]),
}
for k,v in cases.items():
    r = _sqrt_decay(v); print(f"  {k:22s} sqrt={np.sqrt(v)[0]:+.6e}  _sqrt_decay={r[0]:+.6e}")
print()
print("== the FLIP is scale-relative: a SPECTRUM with a big max can flip a near-cutoff EVANESCENT mode ==")
# spectrum: one big evanescent mode (scale) + one tiny near-cutoff POSITIVE-real lam^2 with Im<0 noise
for big in (1.0, 1e2, 1e4):
    x = np.array([big**2 + 0j, (0.5e-8*big)**2 - 1e-30j])
    r = _sqrt_decay(x)
    print(f"  scale={big:9.1e}: lam2[1]={x[1]:.3e} -> lam[1]={r[1]:+.6e}  (Re<0 => |exp(-lam k0 L)|>1)")
    k0L = 1.1424e7
    print(f"      |X| at k0L=1.1424e7 : {np.exp(-r[1].real*k0L):.6e}")
print()
print("== continuity of the forward answer across the band (theta sweep near normal) ==")
from lumenairy.elements.pmm import pmm_jones_2d
def iso(e): return e*np.eye(3,dtype=complex)
S=12; cell=np.zeros((S,S,3,3),dtype=complex); cell[...]=iso(2.25); cell[3:9,3:9]=iso(2.2500000001)
th = np.linspace(0.0, 8e-6, 21)
vals=[]
for t in th:
    o,R,T,J = pmm_jones_2d(0.9e-6,0.9e-6,cell,1.5,1.0,0.4e-6,1.0e-6,theta=float(t),phi=0.0,degree=5,n_orders=3,symmetry=False)
    vals.append(J[0,0])
vals=np.array(vals)
d1=np.diff(vals); print("  |dJxx| step stats: max=%.3e median=%.3e" % (np.max(np.abs(d1)), np.median(np.abs(d1))))
print("  ratio max/median =", np.max(np.abs(d1))/max(np.median(np.abs(d1)),1e-300))
