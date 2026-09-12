from common import *
import cProfile, pstats, io, time, tracemalloc
from lumenairy.elements.pmm import pmm_jones_2d, PMM2DStackHybrid
wl, Px, Py = 1.0e-6, 0.9e-6, 0.9e-6
nsup, nsub = 1.0, 1.45
S = 16
def iso(e): return e*np.eye(3, dtype=complex)
cell = np.zeros((S,S,3,3), dtype=complex); cell[...] = iso(1.0); cell[4:12,4:12] = iso(12.11)

print("== single-layer pmm_jones_2d: wall time vs truncation ==")
for nn, deg in ((5,11),(10,11)):
    for sym in ("auto", False):
        t=time.time(); pmm_jones_2d(Px,Py,cell,nsub,nsup,0.3e-6,wl,degree=deg,n_orders=nn,symmetry=sym); dt=time.time()-t
        print(f"  n_orders={nn:3d} (Nf={(2*nn+1)**2:4d}) degree={deg} symmetry={str(sym):5s}: {dt:6.2f}s", flush=True)

print()
print("== 2-LAYER stack of IDENTICAL layers: is the eigensolve shared? ==")
for nlay, label in ((1,"1 layer"), (2,"2 identical"), (2,"2 distinct")):
    st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=10)
    st.add_layer(0.3e-6, eps_tensor_cell=cell)
    if nlay == 2:
        c2 = cell if label=="2 identical" else cell*1.0001
        st.add_layer(0.3e-6, eps_tensor_cell=c2)
    st.set_source(wl, theta=0.0, phi=0.0)
    t=time.time(); st.solve(); dt=time.time()-t
    print(f"  {label:12s}: {dt:6.2f}s  eig-cache {st._eig_cache.stats()}", flush=True)

print()
print("== profile: single 2-layer solve at n_orders=10 (Nf=441) ==")
st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=10)
st.add_layer(0.3e-6, eps_tensor_cell=cell); st.add_layer(0.25e-6, eps_tensor_cell=cell*1.0001)
st.set_source(wl, theta=0.0, phi=0.0)
pr = cProfile.Profile(); pr.enable(); st.solve(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
print(s.getvalue()[:4200], flush=True)

print()
print("== _tensor_layer_modes SEPARABLE branch: inv() of a DIAGONAL mass matrix? ==")
from lumenairy.elements.pmm.twod import _build_axis
import numpy.linalg as la
ax = _build_axis(Px, [0.25*Px, 0.75*Px], 11, [1,1,1], False)
M = ax["M"]
print("  M shape", M.shape, " is diagonal:", bool(np.allclose(M, np.diag(np.diag(M)))))
t=time.time()
for _ in range(200): la.inv(M)
t1=time.time()-t
t=time.time()
for _ in range(200): d = 1.0/np.diag(M)
t2=time.time()-t
print(f"  200x np.linalg.inv(M): {t1*1e3:.1f} ms ;  200x 1/diag(M): {t2*1e3:.3f} ms  -> {t1/max(t2,1e-12):.0f}x")

print()
print("== memory: LayerCache footprint over a wavelength sweep ==")
st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=6)
st.add_layer(0.3e-6, eps_tensor_cell=cell)
for i, w in enumerate(np.linspace(0.9e-6, 1.1e-6, 6)):
    st.set_source(float(w), theta=0.0, phi=0.0); st.solve()
print("  geom cache:", st._geom_cache.stats(), " budget", st._geom_cache.budget())
print("  eig  cache:", st._eig_cache.stats(),  " budget", st._eig_cache.budget())
