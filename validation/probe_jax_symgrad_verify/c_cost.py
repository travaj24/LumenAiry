"""C: cost RATIOS of the rule, in ONE process, rule ON (default) vs OFF
(_EIG_CLUSTER_GAP_REL = 0 at trace time: the plain composition, i.e. the
pre-fix reverse pass), interleaved repeats, best of N.  jitted gradient
(compile = first call, run = best of 9), eager gradient (best of 3),
vmapped jitted gradient over 4 points, jitted forward."""
import time

from _fix import pmm1d_f, rcwa_f
from _v import dump, jax, jnp, np

import lumenairy.elements.rcwa._core as RC

GAP0 = RC._EIG_CLUSTER_GAP_REL
CASES = {
    "rcwa_c4v_cluster": (rcwa_f("x", "te", n_orders=2), 0.0),
    "rcwa_conical_nocluster": (rcwa_f("corner", "te", n_orders=2, theta=0.2,
                                      phi=0.3), 0.0),
    "pmm1d_normal_cluster": (pmm1d_f("te"), 0.0),
    "pmm1d_0.2_nocluster": (pmm1d_f("te"), 0.2),
}


def build(f, x0, gap, kind):
    RC._EIG_CLUSTER_GAP_REL = gap
    try:
        n = np.asarray(f(x0, np)).size
        w = jnp.asarray(np.random.default_rng(3).standard_normal(n))

        def L(t):
            return jnp.dot(w, f(t, jnp))
        if kind == "grad":
            fn = jax.jit(jax.grad(L))
            arg = jnp.asarray(x0)
        elif kind == "vmap":
            fn = jax.jit(jax.vmap(jax.grad(L)))
            arg = x0 + jnp.asarray([0.0, 1e-3, 2e-3, 3e-3])
        elif kind == "fwd":
            fn = jax.jit(L)
            arg = jnp.asarray(x0)
        elif kind == "eager":
            fn = jax.grad(L)
            arg = jnp.asarray(x0)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))          # compile (+ first run)
        comp = time.perf_counter() - t0
    finally:
        RC._EIG_CLUSTER_GAP_REL = GAP0
    return fn, arg, comp


def timeit(fn, arg, gap):
    RC._EIG_CLUSTER_GAP_REL = gap    # eager calls trace on every call
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        return time.perf_counter() - t0
    finally:
        RC._EIG_CLUSTER_GAP_REL = GAP0


out = {}
for name, (f, x0) in CASES.items():
    for kind, reps in (("grad", 9), ("vmap", 5), ("fwd", 9), ("eager", 3)):
        fon, aon, con = build(f, x0, GAP0, kind)
        foff, aoff, coff = build(f, x0, 0.0, kind)
        ton, toff = [], []
        for _ in range(reps):
            ton.append(timeit(fon, aon, GAP0))
            toff.append(timeit(foff, aoff, 0.0))
        rec = {"run_on": min(ton), "run_off": min(toff),
               "run_ratio": min(ton) / min(toff),
               "first_call_on": con, "first_call_off": coff,
               "first_call_ratio": con / coff}
        out[f"{name}_{kind}"] = rec
        print(name, kind, {k: f"{v:.3g}" for k, v in rec.items()},
              flush=True)
dump("c_cost", out)
