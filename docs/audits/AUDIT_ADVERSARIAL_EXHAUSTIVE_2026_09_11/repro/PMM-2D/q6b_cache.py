from common import *
from lumenairy.elements.pmm import PMM2DStackHybrid
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
S = 12
cell = np.full((S,S), 1.0+0j); cell[3:9,3:9] = 12.25
def mk(**kw):
    d = dict(n_superstrate=nsup, n_substrate=nsub, degree=7, n_orders=3)
    d.update(kw)
    st = PMM2DStackHybrid(Px, Py, **d)
    st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0)
    return st
print("== per-attribute STALE-HIT test: solve, mutate ONE public attr, re-solve, vs a FRESH object ==")
for attr, v0, v1 in (("formulation","li","laurent"),
                     ("symmetry", "auto", False),
                     ("cascade", "fast", "tree"),
                     ("degree", 7, 11),
                     ("grade", False, True),
                     ("n_orders", 3, 4),
                     ("period_x", Px, Px*1.05)):
    try:
        a = mk(**{attr: v0}) if attr in ("formulation","symmetry","cascade","degree","grade","n_orders") else mk()
        _ = a.solve()
        setattr(a, attr, v1)
        reused = a.solve()
        b = mk(**{attr: v1}) if attr in ("formulation","symmetry","cascade","degree","grade","n_orders") else None
        if b is None:
            b = PMM2DStackHybrid(v1, Py, n_superstrate=nsup, n_substrate=nsub, degree=7, n_orders=3)
            b.add_layer(dep, eps_cell=cell); b.set_source(wl, theta=0.0, phi=0.0)
        fresh = b.solve()
        d = np.max(np.abs(np.asarray(reused[3]) - np.asarray(fresh[3])))
        base = mk(**{attr: v0}) if attr in ("formulation","symmetry","cascade","degree","grade","n_orders") else mk()
        bres = base.solve()
        sens = np.max(np.abs(np.asarray(fresh[3]) - np.asarray(bres[3])))
        flag = "STALE" if d > 1e-10 and sens > 1e-10 else ("ok" if d <= 1e-10 else "ok(insensitive)")
        print(f"  {attr:12s} {v0!r:7s}->{v1!r:8s}: |reused-fresh|={d:.3e}  |fresh-base|={sens:.3e}  {flag}", flush=True)
    except Exception as e:
        print(f"  {attr}: {type(e).__name__} {str(e)[:150]}", flush=True)

print()
print("== the same for a TENSOR layer (symmetry drives block_eig there) ==")
t = np.zeros((S,S,3,3), dtype=complex); t[...] = 2.25*np.eye(3)
t[3:9,3:9] = np.array([[4.0,0,0.6],[0,3.4,0],[0.6,0,3.9]], dtype=complex)
def mkt(sym):
    st = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=3,symmetry=sym)
    st.add_layer(dep, eps_tensor_cell=t); st.set_source(wl, theta=0.0, phi=0.0); return st
a = mkt(True); _=a.solve(); a.symmetry=False; reused=a.solve()
fresh = mkt(False).solve(); basе = mkt(True).solve()
print(f"  symmetry True->False (tensor OOP): |reused-fresh|={np.max(np.abs(reused[3]-fresh[3])):.3e}"
      f"  |fresh-base|={np.max(np.abs(np.asarray(fresh[3])-np.asarray(basе[3]))):.3e}")
print("  -> 'symmetry' is NOT in _mode_key; the reuse serves the sym=True modal set.")
