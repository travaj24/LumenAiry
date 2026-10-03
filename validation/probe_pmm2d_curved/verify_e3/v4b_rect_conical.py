"""V4b: the rectangle WIDTH at oblique / conical incidence -- why does the
twin's gradient differ from FD(numpy) by 4.6e-3 at M = 3 (v4_ladder) while
the h^2 premise of the NumPy ladder FAILS?

    python v4b_rect_conical.py M

1. value parity at the reference (twin vs NumPy) at normal / oblique /
   conical incidence: the twin of a Rect stack runs the MAPPED (identity
   transfinite map) route, NumPy the UNMAPPED one;
2. NumPy T(w) on a fine grid around w0: smooth or stepped?  (the unmapped
   far-field / incident node counts are sized from the segment widths);
3. the twin with geometry='static' (the unmapped route, no traced walls)
   vs NumPy;
4. FD(numpy) on the identity-map route (Rect forced through the quadrature
   path by an explicit identity TransfiniteMap on the moving walls).
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, amax, dump, jax, jnp, ladder, np

from lumenairy.elements.pmm import Rect
from lumenairy.elements.pmm._curvemap import TransfiniteMap
from lumenairy.elements.pmm._jax_twod_staggered import StagJaxTwin

M = int(sys.argv[1])
W0 = 0.47
out = {"M": M, "w0": W0}


def build(w, src, backend="numpy", idmap=False):
    if idmap:
        xw = np.array([0.0, 0.6 - w / 2, 0.6 + w / 2, P])
        yw = np.array([0.0, 0.55 - 0.21, 0.55 + 0.21, P])
        cm = TransfiniteMap(xw, yw)
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=2, backend=backend, cmap=cm)
        c = np.ones((3, 3), complex)
        c[1, 1] = 3.6
        st.add_layer(0.4, eps_cell=c)
    else:
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=2, backend=backend)
        st.add_layer(0.4, shapes=[Rect(0.6, 0.55, w, 0.42, 3.6)],
                     background_eps=1.0)
    st.set_source(WL, **src)
    return st


def qn(st):
    o, R, T, J = st.solve()
    return np.array([R[0, 12], T[0, 12], T[1, 12]])


SRC = {"normal": {}, "oblique": dict(theta=0.3),
       "conical": dict(theta=0.3, phi=0.45)}
for tag, src in SRC.items():
    r = {}
    st = build(W0, src, "jax")
    _o, R, T, J = jax.jit(lambda: st.solve()[1:])() if False else \
        (None,) + tuple(np.asarray(a) for a in st.jax_twin().solve()[1:])
    o, Rn, Tn, Jn = build(W0, src).solve()
    r["twin_auto_vs_numpy"] = [amax(R, Rn), amax(T, Tn), amax(J, Jn)]
    tws = StagJaxTwin(build(W0, src), geometry="static")
    _o, Rs, Ts, Js = (np.asarray(a) if a is not None else a
                      for a in tws.solve())
    r["twin_static_vs_numpy"] = [amax(Rs, Rn), amax(Ts, Tn), amax(Js, Jn)]
    o, Ri, Ti, Ji = build(W0, src, idmap=True).solve()
    r["numpy_idmap_vs_numpy_unmapped"] = [amax(Ri, Rn), amax(Ti, Tn),
                                          amax(Ji, Jn)]
    r["twin_auto_vs_numpy_idmap"] = [amax(R, Ri), amax(T, Ti), amax(J, Ji)]
    # fine scan of the NumPy (unmapped) T00 around w0
    ws = W0 + np.linspace(-2e-3, 2e-3, 21) * P
    vals = np.array([qn(build(w, src)) for w in ws])
    d2 = np.diff(vals, 2, axis=0)
    r["numpy_scan_second_diff_max"] = float(np.max(np.abs(d2)))
    r["numpy_scan_second_diff_med"] = float(np.median(np.abs(d2)))
    valsi = np.array([qn(build(w, src, idmap=True)) for w in ws])
    d2i = np.diff(valsi, 2, axis=0)
    r["idmap_scan_second_diff_max"] = float(np.max(np.abs(d2i)))
    r["idmap_scan_second_diff_med"] = float(np.median(np.abs(d2i)))
    # gradients
    tw = st.jax_twin()

    def f(w, tw=tw, st=st):
        p = tw.params()
        p["layers"][0]["shapes"] = [Rect(0.6, 0.55, w, 0.42, 3.6)]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, 12], T[0, 12], T[1, 12]])
    g = np.asarray(jax.jit(jax.jacrev(f))(W0))
    _r, fdn, _c, ratn = ladder(lambda w: qn(build(w, src)), W0,
                               [1e-3, 3e-4, 1e-4], P)
    _r, fdi, _c, rati = ladder(lambda w: qn(build(w, src, idmap=True)), W0,
                               [1e-3, 3e-4, 1e-4], P)
    sc = float(np.max(np.abs(fdi)))
    r.update(AD=g.tolist(), FD_numpy_unmapped=fdn.tolist(),
             FD_numpy_idmap=fdi.tolist(),
             ratios_unmapped=np.asarray(ratn).ravel().tolist(),
             ratios_idmap=np.asarray(rati).ravel().tolist(),
             AD_vs_FD_unmapped=float(np.max(np.abs(g - fdn))) / sc,
             AD_vs_FD_idmap=float(np.max(np.abs(g - fdi))) / sc)
    out[tag] = r
    print(tag, {k: v for k, v in r.items() if "ratios" not in k
                and k not in ("AD", "FD_numpy_unmapped", "FD_numpy_idmap")},
          flush=True)
print(dump(f"v4b_rect_conical_M{M}.json", out))
