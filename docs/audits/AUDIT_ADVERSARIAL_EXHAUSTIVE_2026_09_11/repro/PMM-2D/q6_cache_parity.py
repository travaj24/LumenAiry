from common import *
from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
S = 12
def iso(e): return e*np.eye(3, dtype=complex)

print("== A) _mode_key completeness: is `symmetry` in the key? (tensor layer, block_eig) ==")
# an OUT-OF-PLANE tensor layer is where `symmetry` selects the parity-sign block eig
t = np.zeros((S,S,3,3), dtype=complex)
t[...] = iso(2.25)
blk = np.array([[4.0,0.0,0.6],[0.0,3.4,0.0],[0.6,0.0,3.9]], dtype=complex)
t[3:9,3:9] = blk                                   # out-of-plane, mirror-symmetric layout
def run(sym, reuse=None):
    st = reuse
    if st is None:
        st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=3,symmetry=sym)
        st.add_layer(dep, eps_tensor_cell=t)
    st.symmetry = sym
    st.set_source(wl, theta=0.0, phi=0.0)
    return st, st.solve()
st1, r_true_T = run(True)
st2, r_false  = run(False)
print("  fresh objects: max|J(sym=True) - J(sym=False)| =", np.max(np.abs(r_true_T[3]-r_false[3])))
# now REUSE one object: solve with symmetry=True, then flip to False and re-solve
st3, a = run(True)
st3.symmetry = False
b = st3.solve()
print("  SAME object, symmetry True->False: max|J_b - J(fresh sym=False)| =",
      np.max(np.abs(b[3]-r_false[3])),
      " max|J_b - J_a| =", np.max(np.abs(b[3]-a[3])))
print("  eig-cache stats:", st3.cache_stats() if hasattr(st3,'cache_stats') else st3._eig_cache.stats(), flush=True)

print()
print("== B) other mutable solver attributes vs the keys ==")
st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=3)
cell = np.full((S,S), 1.0+0j); cell[3:9,3:9] = 12.25
st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0)
base = st.solve()
for attr, newv in (("formulation","laurent"), ("cascade","tree"), ("symmetry", False)):
    try:
        st2b = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=3)
        st2b.add_layer(dep, eps_cell=cell); st2b.set_source(wl, theta=0.0, phi=0.0)
        setattr(st2b, attr, newv); fresh = st2b.solve()
        setattr(st, attr, newv); reused = st.solve()
        print(f"  {attr} -> {newv!r}: max|reused - fresh| = {np.max(np.abs(reused[3]-fresh[3])):.3e}"
              f"   (max|fresh - base| = {np.max(np.abs(np.asarray(fresh[3])-np.asarray(base[3]))):.3e})", flush=True)
    except Exception as e:
        print(f"  {attr}: {type(e).__name__} {str(e)[:120]}", flush=True)

print()
print("== C) stack2d (hybrid) vs stack2d_pure: same 2-layer stack ==")
st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=9)
st.add_layer(dep, eps_cell=cell); st.add_layer(0.2e-6, eps=2.25)
st.set_source(wl, theta=0.0, phi=0.0)
oh,Rh,Th,Jh = st.solve()
print(f"  hybrid n_orders=9: E={Rh.sum(1)[0]+Th.sum(1)[0]:.10f}  Jxx={Jh[0,0]:.8f}", flush=True)
for M in (5,7,9):
    try:
        sp = PMM2DStackPure(Px,Py,n_superstrate=nsup,n_substrate=nsub,n_modes=M,n_orders=3)
        sp.add_layer(dep, eps_cell=cell); sp.add_layer(0.2e-6, eps=2.25)
        sp.set_source(wl, theta=0.0, phi=0.0)
        op,Rp,Tp,Jp = sp.solve()
        print(f"  pure   n_modes={M}: E={Rp.sum(1)[0]+Tp.sum(1)[0]:.12f}  Jxx={Jp[0,0]:.8f}", flush=True)
    except Exception as e:
        print(f"  pure M={M}: {type(e).__name__} {str(e)[:200]}", flush=True)

print()
print("== D) thread-safety of the shared LayerCache under solve_vs_wavelength clones ==")
import copy
st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=3)
st.add_layer(dep, eps_cell=cell)
c2 = copy.copy(st)
print("  copy.copy(st)._geom_cache is st._geom_cache :", c2._geom_cache is st._geom_cache)
print("  copy.copy(st)._eig_cache  is st._eig_cache  :", c2._eig_cache is st._eig_cache)
print("  LayerCache has a lock:", hasattr(st._geom_cache, "_lock"))
