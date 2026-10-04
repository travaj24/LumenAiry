"""F1: the library switch (``lumenairy.backend.jax_cluster_rule``) and the
cost of a VMAPPED gradient, rule ON (with the batch-reduced predicate) and
OFF, at a no-cluster point (C2v rectangle, theta 0.2) and at the four-fold
symmetric cell; gradients vs the sequential jitted gradient and ON vs OFF.

    python f1_vmap_switch.py
"""
import time

from _h import dump, jax, jnp, np

from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.rcwa import rcwa_efficiency_2d

base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
BASE, DIRN = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3))
RECT = np.ones((15, 15))
RECT[3:12, 5:10] = 3.2
W = np.random.default_rng(3).uniform(0.5, 1.5, 49)


def rc(cell, **kw):
    def f(t):
        eps = (jnp.asarray(cell) + t * jnp.asarray(DIRN)).astype(complex)
        _o, R, T = rcwa_efficiency_2d(1.2, 1.2, eps, 1.45, 1.0, 0.45, 1.0,
                                      n_orders_x=3, n_orders_y=3, **kw)
        return jnp.sum(jnp.asarray(W) * R) + 0.3 * jnp.sum(jnp.asarray(W) * T)
    return f


def timed(fn, x):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(x))
    comp = time.perf_counter() - t0
    b = np.inf
    for _ in range(7):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        b = min(b, time.perf_counter() - t0)
    return comp, b


out = {}
for name, f, ts in (("rect_theta0.2", rc(RECT, theta=0.2),
                     jnp.asarray([0.0, 0.01, 0.02, 0.03])),
                    ("sym_cell", rc(BASE),
                     jnp.asarray([0.0, 1e-3, 2e-3, 3e-3]))):
    rec = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            fv = jax.jit(jax.vmap(jax.grad(f)))
            fs = jax.jit(jax.grad(f))
            comp, best = timed(fv, ts)
            g = np.asarray(fv(ts))
            seq = np.asarray([fs(t) for t in ts])
            _c1, best1 = timed(fs, ts[0])
        rec["on" if on else "off"] = {
            "vmap4_compile_s": comp, "vmap4_s": best, "single_s": best1,
            "grad": g.tolist(), "vmap_vs_seq": float(np.max(np.abs(g - seq)))}
        print(name, "rule", on, {k: v for k, v in rec[
            "on" if on else "off"].items() if k != "grad"}, flush=True)
    a, b = np.asarray(rec["on"]["grad"]), np.asarray(rec["off"]["grad"])
    rec["on_vs_off_rel"] = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print(name, "on vs off rel %.2e" % rec["on_vs_off_rel"], flush=True)
    out[name] = rec
print(dump("f1_vmap_switch.json", out))
