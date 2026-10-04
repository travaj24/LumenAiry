"""Claims (4) and (5): the RCWAStack cache leak (run with LUMROOT on the base
tree too), and the switch (env var timing, jit keeps its setting, vmap with
the switch off on a mixed batch, the eig count of a vmapped multi-layer
gradient)."""
import json
import os
import subprocess
import sys

from _vc import BUILD, ROOT, fd_rich, jax, jnp, np, rel

out = {"build": BUILD, "root": ROOT}

# ------------------------------------------------ (4) the cache leak
import lumenairy.elements.rcwa._core as RC  # noqa: E402
from lumenairy.elements.rcwa import RCWAStack  # noqa: E402

S = 24
CROSS = np.ones((S, S), complex)
CROSS[8:16, :] = 0.0
CROSS[:, 8:16] = 0.0
ARMX = np.zeros((S, S))
ARMX[8:16, 16:24] = 1.0
BASE = np.where(CROSS == 0.0, 6.25 + 0j, 1.44 + 0j)


def stack_f(t, theta=0.1):
    st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=3, n_orders_y=3)
    st.add_layer(0.2, eps_cell=jnp.asarray(BASE) + t * jnp.asarray(ARMX))
    st.set_source(1.0, theta=theta, phi=0.0)
    return jnp.sum(st.solve().efficiencies()[1])


leak = {}
RC._clear_rcwa_caches()
try:
    v = float(jax.jit(stack_f)(1e-2))
    leak["jit"] = v
    try:
        leak["jit_grad"] = float(jax.jit(jax.grad(stack_f))(1e-2))
    except Exception as e:  # noqa: BLE001
        leak["jit_grad"] = f"{type(e).__name__}"
    try:
        leak["eager"] = float(stack_f(1e-2))
        leak["eager_minus_jit"] = abs(leak["eager"] - v)
    except Exception as e:  # noqa: BLE001
        leak["eager"] = f"{type(e).__name__}"
    leak["cache_len"] = len(RC._HOMOG_CACHE)
    leak["cache_has_tracer"] = any(
        isinstance(a, jax.core.Tracer) for val in RC._HOMOG_CACHE.values()
        for a in (val if isinstance(val, tuple) else (val,)))
except Exception as e:  # noqa: BLE001
    leak["setup"] = f"{type(e).__name__}: {e}"[:200]
out["cache_leak"] = leak
print("leak", leak, flush=True)

import lumenairy.backend as _B  # noqa: E402

if not hasattr(_B, "jax_cluster_rule"):
    json.dump(out, open(f"p5_switch_{BUILD}_base.json", "w"), indent=1)
    sys.exit(0)
from lumenairy.backend import jax_cluster_rule, set_jax_cluster_rule  # noqa: E402

# ------------------------------------------------ (5) env var timing
code = ("import os,sys;sys.path.insert(0,{r!r});{pre}"
        "import lumenairy.backend as B;{post}"
        "print(B.jax_cluster_rule_enabled())")
env_rows = {}
for label, pre, post in (
        ("env_before_import", "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='0';",
         ""),
        ("env_after_import", "",
         "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='0';"),
        ("env_OFF_word", "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='OFF';",
         ""),
        ("env_typo_disable", "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']="
         "'disabled';", "")):
    env = {k: v for k, v in os.environ.items()
           if k != "LUMENAIRY_JAX_CLUSTER_RULE"}
    r = subprocess.run([sys.executable, "-c", code.format(r=ROOT, pre=pre,
                                                          post=post)],
                       capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL, timeout=300)
    env_rows[label] = (r.stdout.strip().splitlines() or [r.stderr[-200:]])[-1]
out["env"] = env_rows
print("env", env_rows, flush=True)


# ------------------------------------------------ jit keeps its setting
from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402

X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def berr(d, xp):
    e = (3.0 + 0.1j) * xp.eye(3, dtype=complex) + d * xp.asarray(X)
    if xp is jnp:
        R, T, Jr, _ = berreman_jones_1d(
            [(e, jnp.asarray(0.25e-6))], jnp.asarray(1.5 + 0j),
            jnp.asarray(1.0 + 0j), jnp.asarray(0.9e-6), angle=0.3)
    else:
        R, T, Jr, _ = berreman_jones_1d([(e, 0.25e-6)], 1.5, 1.0, 0.9e-6,
                                        angle=0.3)
    Jr = xp.ravel(Jr)
    return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(Jr),
                           xp.imag(Jr)])


fd0, ratio, ex = fd_rich(lambda d: berr(d, np), 0.0)
g_on_c = jax.jit(jax.jacrev(lambda d: berr(d, jnp)))
with jax_cluster_rule(True):
    e_on = rel(np.asarray(g_on_c(0.0)), fd0)
with jax_cluster_rule(False):
    e_on_called_off = rel(np.asarray(g_on_c(0.0)), fd0)   # cached: stays ON
    e_on_retrace_off = rel(np.asarray(g_on_c(jnp.asarray(0.0, jnp.float32))
                                      .astype(np.float64)), fd0)
    e_eager_off = rel(np.asarray(jax.jacrev(lambda d: berr(d, jnp))(0.0)),
                      fd0)
out["jit_keeps"] = dict(on=e_on, compiled_on_called_off=e_on_called_off,
                        same_fn_retraced_new_dtype_under_off=e_on_retrace_off,
                        eager_off=e_eager_off,
                        premise=bool(np.all(np.abs(ratio / ex - 1) < 0.12)))
print("jit", out["jit_keeps"], flush=True)


# ------------------------------------------------ vmap, switch off, mixed
def loss(d):
    return jnp.sum(berr(d, jnp) * jnp.arange(1.0, 13.0))


ds = jnp.asarray([0.0, 1e-2, 2e-2])
vm = {}
for on in (True, False):
    with jax_cluster_rule(on):
        gv = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(ds))
        gs = np.asarray([jax.jit(jax.grad(loss))(d) for d in ds])
    fdl = [float(np.sum(fd_rich(lambda d: berr(d, np), float(d))[0]
                        * np.arange(1.0, 13.0))) for d in ds]
    vm[str(on)] = dict(vmap_vs_pointwise=float(np.max(np.abs(gv - gs))),
                       vmap_vs_fd=[abs(a - b) / max(abs(b), 1e-300)
                                   for a, b in zip(gv, fdl)])
out["vmap_mixed"] = vm
print("vmap", vm, flush=True)


# ------------------------------------------------ eig count, 2 layers
def _count_eig(jaxpr):
    import jax.extend.core  # noqa: F401
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "eig":
            n += 1
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(sub, jax.extend.core.ClosedJaxpr):
                    n += _count_eig(sub.jaxpr)
                elif isinstance(sub, jax.extend.core.Jaxpr):
                    n += _count_eig(sub)
    return n


def stack2(t):
    st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=3, n_orders_y=3)
    st.add_layer(0.2, eps_cell=jnp.asarray(BASE) + t * jnp.asarray(ARMX))
    st.add_layer(0.1, eps_cell=jnp.asarray(BASE) * 0.5 + t * jnp.asarray(
        ARMX))
    st.set_source(1.0, theta=0.0, phi=0.0)
    return jnp.sum(st.solve().efficiencies()[1])


cnt = {}
for on in (True, False):
    with jax_cluster_rule(on):
        cnt[str(on)] = _count_eig(jax.make_jaxpr(jax.vmap(jax.grad(stack2)))(
            jnp.asarray([0.0, 1e-2])).jaxpr)
        cnt[str(on) + "_fwd_plain"] = _count_eig(jax.make_jaxpr(stack2)(
            0.0).jaxpr)
out["eig_count_two_layers"] = cnt
print("count", cnt, flush=True)
set_jax_cluster_rule(True)
json.dump(out, open(f"p5_switch_{BUILD}.json", "w"), indent=1)
