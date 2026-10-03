"""The READINGS behind the bars of tests/unit/test_pmm2d_staggered_curved_e3.py,
at the unit-test sizes (M = 3, 3 x 3 grids; the stripe at M = 7 on its
2 x 2 grid).  Output e3_unit_readings.json.

    python e3_unit_readings.py
"""
from _e3common import (P, WL, PMM2DStackPure, dump, jax, jnp, np)  # noqa
from lumenairy.elements.pmm import Circle, Rect, pmm_efficiency_1d

out = {}


def stack(layers, M=3, backend="jax"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend=backend)
    for t, shp, bg in layers:
        st.add_layer(t, shapes=shp, background_eps=bg)
    st.set_source(WL)
    return st


def fd(fun, x0, scale=P):
    rows = [(fun(x0 + h * scale) - fun(x0 - h * scale)) / (2 * h * scale)
            for h in (3e-4, 1e-4)]
    return (9 * rows[1] - rows[0]) / 8, abs(rows[1] - rows[0])


# circle radius
st = stack([(0.5, [Circle(0.6, 0.6, 0.36, 4.0)], 1.0)])
tw = st.jax_twin()
p0 = tw.p0


def fT(r, cx=0.6):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(cx, 0.6, r, 4.0)]
    return st.solve(params=p)[2][0, p0]


fj = jax.jit(fT)
g = float(jax.jit(jax.grad(fT))(0.36))
f_, ch = fd(lambda x: float(fj(x)), 0.36)
out["circle_r"] = {"AD": g, "FD": f_, "rel": abs(g - f_) / abs(f_),
                   "fd_rung_change_rel": ch / abs(f_)}
gc = float(jax.jit(jax.grad(lambda c: fT(0.36, c)))(0.6))
out["circle_dcx_centred"] = gc
print(out, flush=True)

# rectangle width vs the NumPy solve
st = stack([(0.4, [Rect(0.6, 0.6, 0.5, 0.4, 4.0)], 1.0)])
tw = st.jax_twin()


def fR(w):
    p = tw.params()
    p["layers"][0]["shapes"] = [Rect(0.6, 0.6, w, 0.4, 4.0)]
    return st.solve(params=p)[1][0, p0]


g = float(jax.jit(jax.grad(fR))(0.5))
f_, ch = fd(lambda w: float(stack([(0.4, [Rect(0.6, 0.6, w, 0.4, 4.0)],
                                    1.0)], backend="numpy").solve()[1][0, p0]),
            0.5)
out["rect_w_vs_numpy"] = {"AD": g, "FD": f_, "rel": abs(g - f_) / abs(f_),
                          "fd_rung_change_rel": ch / abs(f_)}
print(out["rect_w_vs_numpy"], flush=True)

# material and depth
st = stack([(0.5, [Circle(0.6, 0.6, 0.36, 4.0 + 0.2j)], 1.0)])
tw = st.jax_twin()


def fm(v):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, 0.36, v[0] + 1j * v[1])]
    p["layers"][0]["thickness"] = v[2]
    return st.solve(params=p)[2][0, p0]


x0 = np.array([4.0, 0.2, 0.5])
fmj = jax.jit(fm)
g = np.asarray(jax.jit(jax.grad(fm))(jnp.asarray(x0)))
rows = []
for k in range(3):
    def fk(x, k=k):
        e = np.zeros(3)
        e[k] = x - x0[k]
        return float(fmj(jnp.asarray(x0 + e)))
    f_, ch = fd(fk, x0[k], 1.0)
    rows.append({"AD": float(g[k]), "FD": f_, "rel": abs(g[k] - f_) / abs(f_),
                 "fd_rung_change_rel": ch / abs(f_)})
out["eps_re_im_depth"] = rows
print(rows, flush=True)

# the stripe vs the 1-D PMM twin (2 x 2 grid, M = 7)
st = stack([(0.4, [Rect(0.9, 0.6, 0.6, P, 4.0)], 1.0)], M=7)
tw = st.jax_twin()


def f2(v):
    p = tw.params()
    p["layers"][0]["shapes"] = [Rect(0.9, 0.6, 0.6, P, v[0])]
    p["layers"][0]["thickness"] = v[1]
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, p0], R[1, p0]])


g2 = np.asarray(jax.jit(jax.jacrev(f2))(jnp.asarray([4.0, 0.4])))
res = {}
for k, pol in enumerate(("tm", "te")):
    def f1(v, pol=pol):
        o, R, T = pmm_efficiency_1d(P, jnp.sqrt(v[0]), 1.0, 1.45, 1.0, v[1],
                                    0.5, WL, polarization=pol, degree=40,
                                    stabilize=False)
        return R[int(np.where(np.asarray(o) == 0)[0][0])]
    g1 = np.asarray(jax.grad(f1)(jnp.asarray([4.0, 0.4])))
    res[pol] = {"AD_2d": g2[k].tolist(), "AD_1d": g1.tolist(),
                "rel": (np.abs(g2[k] - g1) / np.abs(g1)).tolist()}
out["stripe_M7"] = res
print(res, flush=True)
dump("e3_unit_readings.json", out)
