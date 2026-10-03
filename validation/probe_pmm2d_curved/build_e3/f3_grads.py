"""E3-3 GRADIENTS vs converged central differences, through the PUBLIC API
(``PMM2DStackPure(backend='jax')``, traced shape objects in ``params``).

    python f3_grads.py CASE M [numpy]

CASE: rect_w | circle_r | fillet_r | sine_A | lc_angle | eps | depth |
      circle_r_conical (theta 0.3, phi 0.4)
Quantities per case (real scalars): R00 and T00 of incident E_x, and for
lc_angle also Re / Im of the reflection Jones entry J_xy.
AD: jax.jit(jax.jacrev(f)).  FD(twin): central differences of jax.jit(f)
on the ladder h / scale = 3e-3, 1e-3, 3e-4, 1e-4, 3e-5; converged value =
the Richardson extrapolation of the last two rungs (h ratio 3, an h^2
error); premise recorded: the rung-to-rung changes fall like h^2.
FD(numpy) (argument 'numpy'): the same with the shipped NumPy solve built
at each parameter value (its own wall grid).
Output f3_<case>_M<M>.json.
"""
import sys

from _e3common import WL, P, PMM2DStackPure, dump, jax, jnp, np, tic  # noqa

from lumenairy.elements.pmm import Circle, FilletRect, Rect, SinusoidalWall

CASE = sys.argv[1]
M = int(sys.argv[2])
DO_NP = len(sys.argv) > 3 and sys.argv[3] == "numpy"


def lc(phi, no=1.5, ne=1.8):
    """In-plane uniaxial tensor, director at angle phi from x (traceable)."""
    xp = jnp if isinstance(phi, jax.Array) or hasattr(phi, "aval") else np
    d = xp.stack([xp.cos(phi), xp.sin(phi), 0.0 * phi])
    return (no ** 2 * xp.eye(3) + (ne ** 2 - no ** 2) * xp.outer(d, d)
            ).astype(complex)


CASES = {
    # name: (x0, shapes(x), background, depth(x), scale)
    "rect_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.4, 4.0)], 1.0,
               lambda x: 0.4, P),
    "circle_r": (0.36, lambda x: [Circle(0.6, 0.6, x, 4.0)], 1.0,
                 lambda x: 0.5, P),
    "fillet_r": (0.1, lambda x: [FilletRect(0.6, 0.6, 0.6, 0.5, x, 4.0)],
                 1.0, lambda x: 0.4, P),
    "sine_A": (0.08, lambda x: [SinusoidalWall("x", 0.6, x, eps=2.25)], 1.0,
               lambda x: 0.4, P),
    "lc_angle": (0.55, lambda x: [Circle(0.6, 0.6, 0.36, lc(x))], 2.0,
                 lambda x: 0.5, 1.0),
    "eps": (4.0, lambda x: [Circle(0.6, 0.6, 0.36, x + 0.2j)], 1.0,
            lambda x: 0.5, 1.0),
    "depth": (0.5, lambda x: [Circle(0.6, 0.6, 0.36, 4.0)], 1.0,
              lambda x: x, 1.0),
    "circle_r_conical": (0.36, lambda x: [Circle(0.6, 0.6, x, 4.0)], 1.0,
                         lambda x: 0.5, P),
}
SRC = {"circle_r_conical": dict(theta=0.3, phi=0.4)}
x0, shp, bg, dep, scale = CASES[CASE]


def build(x, backend="numpy"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend=backend)
    st.add_layer(float(dep(x)), shapes=shp(x), background_eps=bg)
    st.set_source(WL, **SRC.get(CASE, {}))
    return st


t = tic()
ST = build(x0, "jax")
tw = ST.jax_twin()
t_tpl = tic() - t
p0 = tw.p0


def quant(R, T, J):
    q = [R[0, p0], T[0, p0]]
    if CASE == "lc_angle":
        q += [jnp.real(J[0, 1]), jnp.imag(J[0, 1])]
    return jnp.stack([jnp.asarray(v) for v in q])


def f(x):
    p = tw.params()
    lp = p["layers"][0]
    lp["shapes"] = shp(x)
    lp["thickness"] = dep(x)
    _o, R, T, J = ST.solve(params=p)
    return quant(R, T, J)


out = {"case": CASE, "M": M, "x0": x0, "template_s": t_tpl}
fj = jax.jit(f)
t = tic()
v0 = np.asarray(fj(x0))
out["jit_fwd_first_s"] = tic() - t
gj = jax.jit(jax.jacrev(f))
t = tic()
g = np.asarray(gj(x0))
out["jit_grad_first_s"] = tic() - t
t = tic()
np.asarray(gj(x0 * (1 + 1e-7)))
out["jit_grad_repeat_s"] = tic() - t
out["value"] = v0.tolist()
out["AD"] = g.tolist()
steps = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
out["steps"] = steps


def ladder(fun):
    rows = []
    for hs in steps:
        h = hs * scale
        rows.append(((np.asarray(fun(x0 + h)) - np.asarray(fun(x0 - h)))
                     / (2 * h)).tolist())
    rows = np.asarray(rows)
    rich = (9.0 * rows[-1] - rows[-2]) / 8.0
    ch = np.max(np.abs(np.diff(rows, axis=0)), axis=1)
    return rows, rich, ch


rows, rich, ch = ladder(fj)
out["FD_twin"] = rows.tolist()
out["FD_twin_richardson"] = rich.tolist()
out["FD_twin_rung_changes"] = ch.tolist()
den = np.maximum(np.abs(rich), 1e-3 * np.max(np.abs(rich)))
out["AD_vs_FD_twin_rel"] = (np.abs(g - rich) / den).tolist()
out["AD_vs_FD_twin_lastrung_rel"] = (np.abs(g - rows[-1]) / den).tolist()
if DO_NP:
    def fn(x):
        _o, R, T, J = build(x).solve()
        q = [R[0, p0], T[0, p0]]
        if CASE == "lc_angle":
            q += [np.real(J[0, 1]), np.imag(J[0, 1])]
        return np.asarray(q)
    rows, rich2, ch2 = ladder(fn)
    out["FD_numpy"] = rows.tolist()
    out["FD_numpy_richardson"] = rich2.tolist()
    out["FD_numpy_rung_changes"] = ch2.tolist()
    out["AD_vs_FD_numpy_rel"] = (np.abs(g - rich2) / den).tolist()
dump(f"f3_{CASE}_M{M}.json", out)
for k in ("AD", "FD_twin_richardson", "AD_vs_FD_twin_rel",
          "FD_twin_rung_changes", "FD_numpy_richardson",
          "AD_vs_FD_numpy_rel", "jit_fwd_first_s", "jit_grad_first_s",
          "jit_grad_repeat_s"):
    if k in out:
        print(k, out[k])
