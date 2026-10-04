"""Claim (6): cost, rule ON vs OFF in ONE process (OFF = the plain
composition, i.e. the gradient without the rule), jitted; with and without a
cluster; vmapped; jitted FORWARD and its compile (claimed unchanged)."""
import json
import statistics
import time

from _vc import BUILD, jax, jnp, np

from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import PMMStack, pmm_jones_1d
from lumenairy.elements.rcwa import rcwa_efficiency_2d

S = 24
CROSS = np.ones((S, S), complex)
CROSS[8:16, :] = 0.0
CROSS[:, 8:16] = 0.0
ARMX = np.zeros((S, S))
ARMX[8:16, 16:24] = 1.0
BASE = np.where(CROSS == 0.0, 6.25 + 0j, 1.44 + 0j)
RND = np.random.default_rng(11).uniform(-1, 1, (S, S))


def rcwa(theta, offset):
    def f(t):
        e = jnp.asarray(BASE + offset * RND) + t * jnp.asarray(ARMX)
        _o, R, T = rcwa_efficiency_2d(1.3, 1.3, e, 1.5, 1.0, 0.3, 1.0,
                                      theta=theta, n_orders_x=3,
                                      n_orders_y=3)
        return jnp.sum(R * jnp.arange(R.shape[0])) + jnp.sum(T)
    return f


def pj(t):
    ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)
    _o, R, T, _J = pmm_jones_1d(1.2, ER, EG, 1.0, 1.45, 0.45, 0.5, 1.0,
                                angle=t, degree=12, stabilize=False)
    return jnp.sum(R * jnp.arange(R.shape[1])) + jnp.sum(T)


def ps(t):
    st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=12)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(1.0, 2.25)])
    st.set_source(1.0e-6, angle=t)
    _o, R, T, _J = st.solve()
    return jnp.sum(R * jnp.arange(R.shape[1])) + jnp.sum(T)


X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def br(d):
    e = 2.56 * jnp.eye(3, dtype=complex) + d * jnp.asarray(X)
    R, T, Jr, _ = berreman_jones_1d([(e, jnp.asarray(0.3e-6))],
                                    jnp.asarray(1.0 + 0j),
                                    jnp.asarray(1.45 + 0j),
                                    jnp.asarray(1e-6))
    return jnp.sum(R) + jnp.sum(jnp.abs(Jr))


def timed(fn, x, n=7):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(x))
    first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t0)
    return first, statistics.median(ts)


cases = [("rcwa_fourfold_cluster", rcwa(0.0, 0.0), 0.0),
         ("rcwa_offset_theta0.2_nocluster", rcwa(0.2, 1e-2), 0.0),
         ("pmm_jones_1d_at_0_cluster", pj, 0.0),
         ("pmm_jones_1d_at_0.2_accidental", pj, 0.2),
         ("pmmstack_2layer_at_0", ps, 0.0),
         ("berreman_iso_cluster", br, 0.0)]
rows = []
for name, f, x in cases:
    r = dict(case=name)
    for on in (True, False):
        with jax_cluster_rule(on):
            jf = jax.jit(lambda y, f=f: f(y))   # a fresh callable (p11)
            jg = jax.jit(jax.grad(f))
            r[f"fwd_first_{on}"], r[f"fwd_{on}"] = timed(jf, x)
            r[f"grad_first_{on}"], r[f"grad_{on}"] = timed(jg, x)
    r["grad_ratio"] = r["grad_True"] / r["grad_False"]
    r["grad_compile_ratio"] = r["grad_first_True"] / r["grad_first_False"]
    r["fwd_ratio"] = r["fwd_True"] / r["fwd_False"]
    r["fwd_compile_ratio"] = r["fwd_first_True"] / r["fwd_first_False"]
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in r.items()}, flush=True)
    rows.append(r)

# vmap of 4, no cluster anywhere / one cluster in the batch
for name, xs, off in (("vmap4_nocluster", [0.1, 0.11, 0.12, 0.13], 1e-2),
                      ("vmap4_one_cluster", [0.0, 0.11, 0.12, 0.13], 0.0)):
    r = dict(case=name)
    f = rcwa(0.0 if off == 0.0 else 0.2, off)
    for on in (True, False):
        with jax_cluster_rule(on):
            jv = jax.jit(jax.vmap(jax.grad(f)))
            r[f"first_{on}"], r[f"run_{on}"] = timed(jv, jnp.asarray(xs))
    r["ratio"] = r["run_True"] / r["run_False"]
    r["compile_ratio"] = r["first_True"] / r["first_False"]
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in r.items()}, flush=True)
    rows.append(r)
json.dump({"build": BUILD, "rows": rows}, open(f"p6_cost_{BUILD}.json", "w"),
          indent=1)
