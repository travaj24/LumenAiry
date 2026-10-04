"""E2: EAGER (un-jitted) gradient cost and the nested-derivative surface of
``rcwa_efficiency_2d`` -- the shape of ``tests/unit/test_v5_10_3_rcwa_2d_autodiff``
(24 x 24 metal square, 9 x 9 orders, theta 1e-3, TM): eager forward and
``jax.grad`` wall time (best of 3 after a warm-up call), whether ``jax.hessian``
and ``jax.vmap(jax.grad)`` run.

    python e2_eager.py
"""
import time

from _h import dump, jax, jnp, np

from lumenairy.elements.rcwa import rcwa_efficiency_2d

S = 24


def cell(eps_hi):
    x = jnp.linspace(-0.5, 0.5, S)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    return jnp.where((jnp.abs(X) < 0.25) & (jnp.abs(Y) < 0.25), eps_hi,
                     2.1 + 0j)


def f(depth):
    _o, _R, T = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell(6.0 + 0.5j), 1.5,
                                   1.0, depth, 0.6e-6, theta=0.001,
                                   n_orders_x=4, n_orders_y=4,
                                   polarization="tm")
    return jnp.real(jnp.sum(T))


def best(fn, n=3):
    fn()                                   # warm the per-op dispatch caches
    b = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        b = min(b, time.perf_counter() - t0)
    return b


out = {}
d0 = jnp.asarray(0.2e-6)
out["fwd_eager_s"] = best(lambda: f(d0))
out["grad_eager_s"] = best(lambda: jax.grad(f)(d0))
out["grad"] = float(jax.grad(f)(d0))
for name, fn in (("hessian", lambda: jax.hessian(f)(d0)),
                 ("vmap_grad", lambda: jax.vmap(jax.grad(f))(
                     jnp.asarray([0.2e-6, 0.21e-6])))):
    t0 = time.perf_counter()
    try:
        v = fn()
        out[name] = {"ok": True, "value": np.asarray(v).tolist(),
                     "s": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                     "s": time.perf_counter() - t0}
    print(name, out[name], flush=True)
print({k: v for k, v in out.items() if k.endswith("_s")}, flush=True)
print(dump("e2_eager.json", out))
