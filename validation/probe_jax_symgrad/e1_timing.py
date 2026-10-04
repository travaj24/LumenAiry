"""E1: what the fix costs.  Compile time (first call of the jitted function)
and steady wall time (best of 7 after the compile) of the gradient and of
the forward, on the PRE tree (SG_TAG=pre) and the fixed one (SG_TAG=post),
at the test sizes:

  rcwa_sym      the four-fold symmetric cell (every layer eigenvalue in a
                pair: the lifted branch runs), d / d t, TE
  rcwa_rect     a C2v rectangle at oblique incidence (theta 0.2, no
                cluster: the plain branch runs), d / d t, TE
  pmm1d_0       pmm_efficiency_1d d / d(angle) at 0 (half-space pairs:
                lifted), TE
  pmm1d_02      the same at 0.2 rad (no cluster: plain), TE
  rcwa_rect_vmap  rcwa_rect's gradient under jax.vmap over 4 values of t
                (the batched predicate turns lax.cond into a select: BOTH
                branches run)

    python e1_timing.py
"""
import time

from _h import dump, jax, jnp, np

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_2d

P, WL = 1.2, 1.0
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
BASE, DIRN = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3))
RECT = np.ones((15, 15))
RECT[3:12, 5:10] = 3.2


def rc(cell, **kw):
    def f(t):
        eps = (jnp.asarray(cell) + t * jnp.asarray(DIRN)).astype(complex)
        _o, R, T = rcwa_efficiency_2d(P, P, eps, 1.45, 1.0, 0.45, WL,
                                      n_orders_x=3, n_orders_y=3, **kw)
        return jnp.sum(R[:3]) + jnp.sum(T[:3])
    return f


def pm(a0):
    def f(t):
        _o, R, T = pmm_efficiency_1d(P, jnp.asarray(2.0 + 0j), 1.0, 1.45,
                                     1.0, 0.45, 0.5, WL, angle=a0 + t,
                                     degree=12, stabilize=False)
        return jnp.sum(R[:3]) + jnp.sum(T[:3])
    return f


CASES = {"rcwa_sym": rc(BASE), "rcwa_rect": rc(RECT, theta=0.2),
         "pmm1d_0": pm(0.0), "pmm1d_02": pm(0.2)}


def timed(fn, x):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(x))
    comp = time.perf_counter() - t0
    best = np.inf
    for _ in range(7):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        best = min(best, time.perf_counter() - t0)
    return comp, best


out = {}
for name, f in CASES.items():
    cg, wg = timed(jax.jit(jax.grad(f)), 0.0)
    cf, wf = timed(jax.jit(f), 0.0)
    out[name] = {"grad_compile_s": cg, "grad_s": wg, "fwd_compile_s": cf,
                 "fwd_s": wf}
    print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
f = CASES["rcwa_rect"]
cg, wg = timed(jax.jit(jax.vmap(jax.grad(f))), jnp.zeros(4))
out["rcwa_rect_vmap4"] = {"grad_compile_s": cg, "grad_s": wg}
print("rcwa_rect_vmap4", out["rcwa_rect_vmap4"], flush=True)
print(dump("e1_timing.json", out))
