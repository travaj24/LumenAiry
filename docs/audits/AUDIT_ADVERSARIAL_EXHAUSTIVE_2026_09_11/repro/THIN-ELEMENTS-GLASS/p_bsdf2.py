import numpy as np, sys, warnings, time
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.bsdf import LambertianBSDF, GaussianBSDF, HarveyShackBSDF, make_bsdf

def tis_fine(b, n=4_000_00):
    """High-accuracy TIS by dense midpoint rule in theta (azimuthally symmetric)."""
    th = (np.arange(n)+0.5)*(np.pi/2)/n
    S = np.stack([np.sin(th), np.zeros(n), np.cos(th)], axis=-1)
    B = np.asarray(b.evaluate(np.array([0.,0.,-1.]), S), dtype=float)
    return float(2*np.pi*np.sum(B*np.cos(th)*np.sin(th))*(np.pi/2)/n)

print("=== TIS: library default 256x128 quadrature vs dense 400k-point reference ===", flush=True)
print(f"{'model':44s} {'TIS(lib)':>12s} {'TIS(ref)':>12s} {'rel err':>10s}", flush=True)
for lbl, b in [
  ("Lambertian(rho=0.5)  [closed form override]", LambertianBSDF(rho=0.5)),
  ("Gaussian(s=1e-2,f=1e-2)[closed form override]", GaussianBSDF(sigma_rad=1e-2, scattered_fraction=1e-2)),
  ("HarveyShack(b0=1,l=1e-1,s=2) [NUMERIC]", HarveyShackBSDF(b0=1.0, l=0.1, s=2.0)),
  ("HarveyShack(b0=1,l=1e-2,s=2) [NUMERIC, DEFAULT l]", HarveyShackBSDF(b0=1.0, l=0.01, s=2.0)),
  ("HarveyShack(b0=1,l=1e-3,s=2) [NUMERIC]", HarveyShackBSDF(b0=1.0, l=0.001, s=2.0)),
  ("HarveyShack(b0=1,l=1e-2,s=1.5)[NUMERIC]", HarveyShackBSDF(b0=1.0, l=0.01, s=1.5)),
  ("HarveyShack DEFAULTS", HarveyShackBSDF()),
]:
    tl = b.total_integrated_scatter(); ta = tis_fine(b)
    print(f"{lbl:44s} {tl:12.6g} {ta:12.6g} {(tl-ta)/ta:+10.2%}", flush=True)
print("\n  analytic TIS for s=2:  pi*b0*l^2*ln(1+1/l^2)", flush=True)
for l in (0.1, 0.01, 0.001):
    b = HarveyShackBSDF(b0=1.0, l=l, s=2.0)
    ana = np.pi*l*l*np.log(1+1/l**2)
    print(f"   l={l:6.4f}: analytic={ana:.6g}  lib={b.total_integrated_scatter():.6g}  ref={tis_fine(b):.6g}"
          f"  lib/analytic={b.total_integrated_scatter()/ana:.4f}", flush=True)

print("\n=== Gaussian oblique TIS from evaluate() ===", flush=True)
b = GaussianBSDF(sigma_rad=0.01, scattered_fraction=0.02)
nt, npz = 3000, 360
th = np.linspace(1e-9, np.pi/2, nt); ph = np.linspace(0,2*np.pi,npz,endpoint=False)
T,Pp = np.meshgrid(th,ph,indexing='ij')
S=np.empty(T.shape+(3,)); S[...,0]=np.sin(T)*np.cos(Pp); S[...,1]=np.sin(T)*np.sin(Pp); S[...,2]=np.cos(T)
for ang in (0,30,60):
    a=np.deg2rad(ang); inc=np.array([np.sin(a),0.,-np.cos(a)])
    B=np.asarray(b.evaluate(inc,S),dtype=float)
    tis=float((B*np.cos(T)*np.sin(T)).sum()*(th[1]-th[0])*(ph[1]-ph[0]))
    print(f"  theta_i={ang:3d} deg  int BSDF cos dOmega = {tis:.6f}  target={b.scattered_fraction}", flush=True)

print("\n=== make_bsdf typo tolerance ===", flush=True)
print("  ", make_bsdf({'kind':'gaussian','sigma':0.001,'scatter_fraction':0.5}), flush=True)
print("  ", make_bsdf({'kind':'harvey_shack','B':0.02,'A':1e-3}), flush=True)

print("\n=== HarveyShack.sample timing / acceptance ===", flush=True)
for l in (0.1,0.01,0.001):
    b=HarveyShackBSDF(b0=1.0,l=l,s=2.0)
    t0=time.perf_counter(); d=b.sample(np.array([0,0,-1.]),20000,rng=1); t1=time.perf_counter()
    st=np.hypot(d[:,0],d[:,1])
    print(f"  l={l:6.4f}: 20k samples {t1-t0:7.3f}s  mean sin(th)={st.mean():.5f}", flush=True)
