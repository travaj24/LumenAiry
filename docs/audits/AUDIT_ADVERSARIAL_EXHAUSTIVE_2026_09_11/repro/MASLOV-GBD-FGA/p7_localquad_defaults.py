import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.lenses_maslov import _integrate_local_quadrature
from lumenairy.elements.lenses import _multi_indices_total_degree

poly_order = 2
mi = _multi_indices_total_degree(4, poly_order)
K1 = np.array([k[0] for k in mi], np.int64); K2 = np.array([k[1] for k in mi], np.int64)
K3 = np.array([k[2] for k in mi], np.int64); K4 = np.array([k[3] for k in mi], np.int64)
idx = {k: j for j, k in enumerate(mi)}
c = 1.0
def make(A,B,Cxy=0.0):
    co = np.zeros(len(mi))
    co[idx[(0,0,0,0)]] += 0.25*A + 0.25*B
    co[idx[(0,0,2,0)]] += 0.25*A
    co[idx[(0,0,0,2)]] += 0.25*B
    co[idx[(0,0,1,1)]] += Cxy            # cross term  Cxy*u3*u4 -> H34 = Cxy
    sx = np.zeros(len(mi)); sx[idx[(0,0,1,0)]] = c
    sy = np.zeros(len(mi)); sy[idx[(0,0,0,1)]] = c
    return co, sx, sy
def sE(a,b): return np.ones_like(a, dtype=np.complex128)
def pg(*a, **k): pass
u0 = np.zeros((1,1)); ib = np.array([True])
def lq(co,sx,sy,ns,ws):
    return _integrate_local_quadrature(co,sx,sy,K1,K2,K3,K4,poly_order,1,u0,u0,ib,
                                       1.0,1.0,sE,30,1e-12,ns,ws,pg,False)[0,0]
def exact(A,B,Cxy=0.0):
    # int exp(i pi (A u3^2 + B u4^2 + 2 Cxy u3 u4)) d2u = e^{i pi sig/4}/sqrt|det|
    H = np.array([[A,Cxy],[Cxy,B]]); ev = np.linalg.eigvalsh(H)
    sig = int(np.sign(ev).sum())
    return np.exp(1j*np.pi*sig/4)/np.sqrt(abs(np.linalg.det(H)))

print("=== DEFAULT local_n_samples=8, local_window_sigma=3.0 ===")
for A,B,C in [(40,40,0),(40,4,0),(4,40,0),(100,10,0),(40,40,30),(60,20,25)]:
    co,sx,sy = make(A,B,C); ex = exact(A,B,C); v = lq(co,sx,sy,8,3.0)
    print(f" A={A:5} B={B:5} H34={C:4}: exact={ex:20.6g} lq={v:20.6g} relerr={abs(v-ex)/abs(ex):8.2e}")

print("\n=== convergence sweep (A=B=40, isotropic) ===")
co,sx,sy = make(40,40); ex = exact(40,40)
for ws in (3.,4.,6.,10.,20.,40.):
    row=[]
    for ns in (8,16,32,64,128):
        v = lq(co,sx,sy,ns,ws); row.append(f"{abs(v-ex)/abs(ex):8.1e}")
    print(f"  window_sigma={ws:5.1f}: " + " ".join(f"n={n}:{r}" for n,r in zip((8,16,32,64,128),row)))

print("\n=== swap check: same |Hessian|, transposed axes (should be identical) ===")
for (A,B) in [(40,4),(4,40),(200,4),(4,200),(1000,10),(10,1000)]:
    co,sx,sy = make(A,B); ex = exact(A,B)
    v3 = lq(co,sx,sy,32,6.0)
    print(f" A={A:6} B={B:6}: relerr(n=32,ws=6) = {abs(v3-ex)/abs(ex):9.3e}")
