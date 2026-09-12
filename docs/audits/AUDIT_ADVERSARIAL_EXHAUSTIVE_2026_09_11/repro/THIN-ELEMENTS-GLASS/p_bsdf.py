import numpy as np, sys, warnings, time
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.bsdf import LambertianBSDF, GaussianBSDF, HarveyShackBSDF, make_bsdf

def tis_accurate(b, n_theta=200001):
    """High-accuracy TIS by 1-D Gauss-Legendre in theta (azimuthally symmetric models)."""
    th, w = np.polynomial.legendre.leggauss(20000)
    th = 0.5*(th+1)*(np.pi/2); w = w*0.5*(np.pi/2)
    S = np.empty((th.size,3)); S[:,0]=np.sin(th); S[:,1]=0.0; S[:,2]=np.cos(th)
    inc = np.array([0.0,0.0,-1.0])
    B = np.asarray(b.evaluate(inc, S), dtype=float)
    return float(2*np.pi*np.sum(w*B*np.cos(th)*np.sin(th)))

print("=== TIS: library default quadrature vs high-accuracy Gauss-Legendre ===")
print(f"{'model':42s} {'TIS(lib)':>12s} {'TIS(GL 20k)':>12s} {'rel err':>10s}")
cases = [
  ("Lambertian(rho=0.5) [closed form]", LambertianBSDF(rho=0.5), 0.5),
  ("Gaussian(sigma=1e-2,f=1e-2) [closed]", GaussianBSDF(sigma_rad=1e-2, scattered_fraction=1e-2), 1e-2),
  ("HarveyShack(b0=1, l=1e-1, s=2) [numeric]", HarveyShackBSDF(b0=1.0, l=0.1, s=2.0), None),
  ("HarveyShack(b0=1, l=1e-2, s=2) [numeric]", HarveyShackBSDF(b0=1.0, l=0.01, s=2.0), None),
  ("HarveyShack(b0=1, l=1e-3, s=2) [numeric]", HarveyShackBSDF(b0=1.0, l=0.001, s=2.0), None),
  ("HarveyShack(b0=1, l=1e-2, s=1.8)[numeric]",HarveyShackBSDF(b0=1.0, l=0.01, s=1.8), None),
  ("HarveyShack DEFAULTS (b0=1,l=1e-2,s=2)",  HarveyShackBSDF(), None),
]
for lbl, b, closed in cases:
    t0=time.perf_counter(); tl = b.total_integrated_scatter(); t1=time.perf_counter()
    ta = tis_accurate(b)
    rel = (tl-ta)/ta if ta else float('nan')
    print(f"{lbl:42s} {tl:12.6g} {ta:12.6g} {rel:+10.2%}   ({t1-t0:.3f}s)")
# analytic TIS for Harvey-Shack s=2: int b0/(1+(sin/l)^2) cos sin dth dphi, u=sin
# = 2pi b0 int_0^1 u/(1+(u/l)^2) du = pi b0 l^2 ln(1+1/l^2)
for l in (0.1, 0.01, 0.001):
    b = HarveyShackBSDF(b0=1.0, l=l, s=2.0)
    ana = np.pi*1.0*l*l*np.log(1+1/l**2)
    print(f"  analytic TIS (s=2, l={l}) = {ana:.6g} ; lib={b.total_integrated_scatter():.6g} ; GL={tis_accurate(b):.6g}")

print("\n=== Gaussian oblique-incidence TIS from evaluate() ===")
b = GaussianBSDF(sigma_rad=0.01, scattered_fraction=0.02)
for ang_deg in (0, 30, 60):
    a=np.deg2rad(ang_deg)
    inc = np.array([np.sin(a), 0.0, -np.cos(a)])
    # 2-D hemisphere quadrature around the specular direction
    nt, npz = 4000, 720
    th = np.linspace(1e-9, np.pi/2, nt); ph = np.linspace(0,2*np.pi,npz,endpoint=False)
    T,Pp = np.meshgrid(th,ph,indexing='ij')
    S=np.empty(T.shape+(3,)); S[...,0]=np.sin(T)*np.cos(Pp); S[...,1]=np.sin(T)*np.sin(Pp); S[...,2]=np.cos(T)
    B=np.asarray(b.evaluate(inc,S),dtype=float)
    tis = float((B*np.cos(T)*np.sin(T)).sum()*(th[1]-th[0])*(ph[1]-ph[0]))
    print(f"  theta_i={ang_deg:3d} deg  int BSDF cos dOmega = {tis:.6f}  (target scattered_fraction={b.scattered_fraction})")

print("\n=== make_bsdf silently drops unknown keys? ===")
m = make_bsdf({'kind':'gaussian','sigma':0.001,'scatter_fraction':0.5})
print("  make_bsdf({'kind':'gaussian','sigma':0.001,'scatter_fraction':0.5}) ->", m)

print("\n=== HarveyShack.sample acceptance efficiency ===")
for l in (0.1, 0.01, 0.001):
    b=HarveyShackBSDF(b0=1.0,l=l,s=2.0)
    t0=time.perf_counter(); d=b.sample(np.array([0,0,-1.0]), 20000, rng=1); t1=time.perf_counter()
    st = np.hypot(d[:,0],d[:,1])
    print(f"  l={l}: 20k samples in {t1-t0:.3f}s ; mean sin(theta)={st.mean():.5f}, median={np.median(st):.5f}")
