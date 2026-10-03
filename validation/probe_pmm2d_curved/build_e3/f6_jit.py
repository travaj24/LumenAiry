"""E3-6 jax.jit OF A FULL SOLVE: compile once, re-run at new radii without
recompiling; timings against the NumPy solve.

    python f6_jit.py M

A Python-side trace counter inside the solved function counts traces (one
per compilation); the jitted function is called at three radii and the
gradient at three radii; the counters and jit's own cache size must stay 1.
Timings (best of 3): jitted forward, jitted gradient, the eager twin
forward, the NumPy solve built at the same radius.
Output f6_jit_M<M>.json.
"""
import sys

from _e3common import WL, P, PMM2DStackPure, dump, jax, jnp, np, tic  # noqa

from lumenairy.elements.pmm import Circle

M = int(sys.argv[1])
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.36, 4.0)], background_eps=1.0)
st.set_source(WL)
t = tic()
tw = st.jax_twin()
t_tpl = tic() - t
p0 = tw.p0
TRACES = {"f": 0}


def f(r):
    TRACES["f"] += 1
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 4.0)]
    _o, R, T, J = st.solve(params=p)
    return T[0, p0]


fj = jax.jit(f)
gj = jax.jit(jax.grad(f))
out = {"M": M, "template_s": t_tpl}
t = tic()
fj(0.36).block_until_ready()
out["fwd_compile_plus_run_s"] = tic() - t
n_after_first = TRACES["f"]
vals = [float(fj(r)) for r in (0.30, 0.33, 0.39)]
out["traces_after_forward_calls"] = TRACES["f"]
out["forward_cache_size"] = fj._cache_size()
t = tic()
gj(0.36).block_until_ready()
out["grad_compile_plus_run_s"] = tic() - t
n_g = TRACES["f"]
grads = [float(gj(r)) for r in (0.30, 0.33, 0.39)]
out["traces_after_grad_calls"] = TRACES["f"]
out["grad_cache_size"] = gj._cache_size()
out["traces_first_forward"] = n_after_first
out["traces_first_grad"] = n_g - out["traces_after_forward_calls"]


def best(fun, n=3):
    ts = []
    for _ in range(n):
        t = tic()
        fun()
        ts.append(tic() - t)
    return min(ts)


out["jit_forward_s"] = best(lambda: fj(0.35).block_until_ready())
out["jit_grad_s"] = best(lambda: gj(0.35).block_until_ready())
out["eager_forward_s"] = best(lambda: f(jnp.asarray(0.35)), n=1)


def numpy_solve():
    s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                       n_modes=M, n_orders=2)
    s.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.35, 4.0)], background_eps=1.0)
    s.set_source(WL)
    s.solve()


out["numpy_solve_s"] = best(numpy_solve, n=2)
out["values"] = vals
out["grads"] = grads
dump(f"f6_jit_M{M}.json", out)
print(out)
