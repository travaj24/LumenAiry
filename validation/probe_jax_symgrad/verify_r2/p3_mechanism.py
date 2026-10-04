"""Claim B: the record / replay mechanism ``rcwa._core._jax_cluster_routed``
on synthetic solves (a matrix exponential through ``_jax_twin_eig``: a
basis-invariant consumer whose gradient at a degenerate pair needs the
in-cluster coupling), plus attempts to desynchronise record and replay."""
import json
import threading

import scipy.linalg as sla
from _vc import BUILD, jax, jnp, np

import lumenairy.elements.rcwa._core as RC
from lumenairy.backend import jax_cluster_rule

out = {"build": BUILD}
rng = np.random.default_rng(5)
N = 5
LAM0 = np.diag([1.0, 1.0, 2.0, 3.5, 3.5]).astype(complex)
Q = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
A0 = Q @ LAM0 @ np.linalg.inv(Q)          # non-normal, two exact pairs
C = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
Wt = rng.standard_normal((N, N))


def expm_eig(A, xp_eig):
    lam, V = xp_eig(A)
    return V @ jnp.diag(jnp.exp(lam)) @ jnp.linalg.inv(V)


def f_np(t):
    return float(np.real(np.sum(sla.expm(A0 + t * C) * Wt)))


def fd(fn, x0=0.0, h=1e-4):
    return (8 * (fn(x0 + h) - fn(x0 - h)) - (fn(x0 + 2 * h)
                                             - fn(x0 - 2 * h))) / (12 * h)


g_true = fd(f_np)


def f_routed(t):
    A = jnp.asarray(A0) + t * jnp.asarray(C)

    def solve():
        return jnp.real(jnp.sum(expm_eig(A, RC._jax_twin_eig) * Wt))
    return RC._jax_cluster_routed(solve)


# (a) the mechanism on the toy: rule on / off
res = {}
for on in (True, False):
    with jax_cluster_rule(on):
        res[on] = float(jax.jit(jax.grad(f_routed))(0.0))
        res[str(on) + "_eager"] = float(jax.grad(f_routed)(0.0))
out["a_toy"] = dict(g_true=g_true, on=res[True], off=res[False],
                    on_eager=res["True_eager"], err_on=abs(res[True] - g_true)
                    / abs(g_true), err_off=abs(res[False] - g_true)
                    / abs(g_true))
print("a", out["a_toy"], flush=True)


# (b) nested routed solves (an inner routed solve inside an outer one)
B0 = np.diag([2.0, 2.0, 0.5]).astype(complex)
CB = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))


def f_nested_np(t):
    return (float(np.real(np.sum(sla.expm(A0 + t * C) * Wt)))
            + float(np.real(np.sum(sla.expm(B0 + t * CB)))) ** 2)


def f_nested(t):
    A = jnp.asarray(A0) + t * jnp.asarray(C)
    B = jnp.asarray(B0) + t * jnp.asarray(CB)

    def inner():
        return jnp.real(jnp.sum(expm_eig(B, RC._jax_twin_eig)))

    def outer():
        x = RC._jax_cluster_routed(inner)
        return jnp.real(jnp.sum(expm_eig(A, RC._jax_twin_eig) * Wt)) + x ** 2
    return RC._jax_cluster_routed(outer)


gn_true = fd(f_nested_np)
gn = float(jax.jit(jax.grad(f_nested))(0.0))
gn_e = float(jax.grad(f_nested)(0.0))
out["b_nested"] = dict(g_true=gn_true, jit=gn, eager=gn_e,
                       err_jit=abs(gn - gn_true) / abs(gn_true),
                       err_eager=abs(gn_e - gn_true) / abs(gn_true),
                       route_after=repr(RC._JAX_TWIN_EIG_ROUTE.get()))
print("b", out["b_nested"], flush=True)


# (c) exception paths leave no route behind
def f_raise_at(k):
    calls = [0]

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)

        def solve():
            calls[0] += 1
            if calls[0] == k:
                raise ValueError(f"boom at pass {k}")
            return jnp.real(jnp.sum(expm_eig(A, RC._jax_twin_eig) * Wt))
        return RC._jax_cluster_routed(solve)
    return f


crows = []
for k in (1, 2, 3):
    err = None
    try:
        jax.jit(jax.grad(f_raise_at(k)))(0.0)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:60]}"
    route = RC._JAX_TWIN_EIG_ROUTE.get()
    after = float(jax.jit(jax.grad(f_routed))(0.0))
    crows.append(dict(raise_at_pass=k, err=err, route_after=repr(route),
                      next_grad_err=abs(after - g_true) / abs(g_true)))
out["c_exceptions"] = crows
print("c", crows, flush=True)


# (d) a solve whose eig COUNT differs between its passes (state outside the
# trace: a call counter), fewer and more on the replay (pass 3)
def f_count(delta):
    calls = [0]

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)

        def solve():
            calls[0] += 1
            n = 2 + (delta if calls[0] == 3 else 0)
            acc = 0.0
            for _ in range(n):
                acc = acc + jnp.real(jnp.sum(expm_eig(A, RC._jax_twin_eig)
                                             * Wt))
            return acc
        return RC._jax_cluster_routed(solve)
    return f


drows = []
for delta in (-1, +1):
    err = None
    val = None
    try:
        val = float(jax.jit(jax.grad(f_count(delta)))(0.0))
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:120]}"
    drows.append(dict(delta=delta, err=err, value=val,
                      route_after=repr(RC._JAX_TWIN_EIG_ROUTE.get())))
out["d_count_mismatch"] = drows
print("d", drows, flush=True)


# (e) same count, same shapes, different ORDER on the replay (pass 3): no
# check ties a replayed eig to its recorded problem
A1 = np.diag([0.3, 0.7, 1.1, 1.9, 2.4]).astype(complex)


def f_order():
    calls = [0]

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)
        B = jnp.asarray(A1)

        def solve():
            calls[0] += 1
            mats = (A, B) if calls[0] != 3 else (B, A)
            ea = expm_eig(mats[0], RC._jax_twin_eig)
            eb = expm_eig(mats[1], RC._jax_twin_eig)
            if calls[0] == 3:
                ea, eb = eb, ea
            return jnp.real(jnp.sum(ea * Wt)) + jnp.real(jnp.trace(eb))
        return RC._jax_cluster_routed(solve)
    return f


def f_order_np(t):
    return (float(np.real(np.sum(sla.expm(A0 + t * C) * Wt)))
            + float(np.real(np.trace(sla.expm(A1)))))


v_true = f_order_np(0.0)
try:
    v_jit = float(jax.jit(f_order())(0.0))
    out["e_reorder"] = dict(v_true=v_true, v_jit=v_jit,
                            rel=abs(v_jit - v_true) / abs(v_true))
except Exception as e:  # noqa: BLE001
    out["e_reorder"] = dict(err=f"{type(e).__name__}: {str(e)[:120]}")
print("e", out["e_reorder"], flush=True)


# (f) threads: 4 threads tracing jit(grad) of routed solves concurrently,
# two of them with the switch flipped by the context manager
tres = {}


def worker(i):
    try:
        g = float(jax.jit(jax.grad(f_routed))(0.0 + 0.0 * i))
        tres[i] = abs(g - g_true) / abs(g_true)
    except Exception as e:  # noqa: BLE001
        tres[i] = f"{type(e).__name__}: {e}"


ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for th in ths:
    th.start()
for th in ths:
    th.join()
out["f_threads"] = tres
print("f", tres, flush=True)

# (g) the switch's context manager interleaved (two scopes entered and left
# out of order -- what two threads each using the context manager do)
from lumenairy.backend import jax_cluster_rule_enabled  # noqa: E402

a, b = jax_cluster_rule(False), jax_cluster_rule(True)
a.__enter__()
b.__enter__()
a.__exit__(None, None, None)
b.__exit__(None, None, None)
out["g_interleaved_ctx_final_state"] = jax_cluster_rule_enabled()
from lumenairy.backend import set_jax_cluster_rule  # noqa: E402

set_jax_cluster_rule(True)
# exception inside the context manager restores
try:
    with jax_cluster_rule(False):
        raise KeyError("x")
except KeyError:
    pass
out["g_ctx_restores_on_exception"] = jax_cluster_rule_enabled()
print("g", out["g_interleaved_ctx_final_state"],
      out["g_ctx_restores_on_exception"], flush=True)


# (h) an eig inside an inner jax.vmap / lax.scan of the solve
def f_inner(kind):
    def f(t):
        As = jnp.stack([jnp.asarray(A0) + t * jnp.asarray(C)] * 2)

        def solve():
            if kind == "vmap":
                Es = jax.vmap(lambda A: expm_eig(A, RC._jax_twin_eig))(As)
            else:
                def body(c, A):
                    return c, expm_eig(A, RC._jax_twin_eig)
                _c, Es = jax.lax.scan(body, 0.0, As)
            return jnp.real(jnp.sum(Es[0] * Wt))
        return RC._jax_cluster_routed(solve)
    return f


hrows = {}
for kind in ("vmap", "scan"):
    try:
        g = float(jax.jit(jax.grad(f_inner(kind)))(0.0))
        hrows[kind] = dict(err=abs(g - g_true) / abs(g_true))
    except Exception as e:  # noqa: BLE001
        hrows[kind] = dict(exc=f"{type(e).__name__}: {str(e)[:100]}")
    hrows[kind]["route_after"] = repr(RC._JAX_TWIN_EIG_ROUTE.get())
out["h_inner_transform"] = hrows
print("h", hrows, flush=True)


# (i) jit with a changed static shape: retrace, both correct
def f_sized(t, n):
    A = jnp.asarray(A0[:n, :n]) + t * jnp.asarray(C[:n, :n])

    def solve():
        return jnp.real(jnp.sum(expm_eig(A, RC._jax_twin_eig) * Wt[:n, :n]))
    return RC._jax_cluster_routed(solve)


gj = jax.jit(jax.grad(f_sized), static_argnums=1)
irows = {}
for n in (5, 2, 5):
    tru = fd(lambda t: float(np.real(np.sum(sla.expm(A0[:n, :n] + t
                                                     * C[:n, :n])
                                            * Wt[:n, :n]))))
    irows.setdefault(str(n), []).append(abs(float(gj(0.0, n)) - tru)
                                        / abs(tru))
out["i_static_shapes"] = irows
print("i", irows, flush=True)

# (j) forward-mode: jvp / jacfwd / hessian of a routed solve
jrows = {}
for name, fn in (("jvp", lambda: jax.jvp(f_routed, (0.0,), (1.0,))),
                 ("jacfwd", lambda: jax.jacfwd(f_routed)(0.0)),
                 ("hessian", lambda: jax.hessian(f_routed)(0.0)),
                 ("grad_of_grad", lambda: jax.grad(jax.grad(f_routed))(0.0))):
    try:
        v = fn()
        jrows[name] = dict(ok=True, value=repr(v)[:80])
    except Exception as e:  # noqa: BLE001
        jrows[name] = dict(ok=False, exc=f"{type(e).__name__}: {str(e)[:140]}")
out["j_forward_mode"] = jrows
print("j", jrows, flush=True)
json.dump(out, open(f"p3_mechanism_{BUILD}.json", "w"), indent=1,
          default=str)
