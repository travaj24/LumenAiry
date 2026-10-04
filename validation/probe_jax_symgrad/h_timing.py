"""H: what the round-2 routing costs on the routed twins.  Compile time (first
call) and steady wall time (best of 7) of jit(grad) and jit(forward) of a
scalar (sum of the d4-d7 outputs), PRE (SG_TAG=pre) vs POST (SG_TAG=r2post).
Context: ``h_eigcount`` shows jit(jacrev) holding 5x the eig custom calls on
POST (compiled count includes both arms of a cond, so it is not an
execution count -- this measures the time).

  berreman_exy_0          d4 exy, d / d(delta) at 0 (exact pairs)
  pmmjones1d_0 / _02      d5, d / d(angle) at 0 (pairs) / 0.2 (none)
  pmmstack_shared_2L_0    d6, d / d(angle) at 0
  stack2d_c4v_laurent_0   d7, d / d(theta) at 0
  stack2d_traced_corner_laurent_0   d7 traced layout, d / d(delta) at 0

    python h_timing.py
"""
import time

from _h import dump, jax, jnp, np

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import PMM2DStackHybrid, PMMStack, pmm_jones_1d

XEXY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def s(*arrs):
    return sum(jnp.sum(jnp.abs(jnp.asarray(a)) ** 2)
               for a in jax.tree_util.tree_leaves(arrs))


def f_bm(d):
    e = 2.56 * jnp.eye(3, dtype=complex) + d * jnp.asarray(XEXY)
    R, T, Jr, _ = berreman_jones_1d(
        [(e, jnp.asarray(0.3e-6)), (jnp.asarray(2.1 + 0j),
                                    jnp.asarray(0.1e-6))],
        jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j), jnp.asarray(1e-6))
    return s(R, T, Jr)


def f_pj(a):
    _o, R, T, J = pmm_jones_1d(1.2, 4.0 * np.eye(3), np.eye(3), 1.0, 1.45,
                               0.45, 0.5, 1.0, angle=a, degree=12,
                               stabilize=False)
    return s(R, T, J)


def f_ps(a):
    st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=12)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.add_layer(0.15e-6, segments=[(0.1, 1.0), (0.3, 2.25), (0.6, 1.0)])
    st.set_source(1.0e-6, angle=a)
    _o, R, T, J = st.solve()
    return s(R, T, J)


C4 = np.full((6, 6), 1.0 + 0j)
C4[0:3, 0:3] = 4.0
LAY = np.zeros((6, 6), dtype=np.int64)
for _i in range(3):
    for _j in range(3):
        LAY[_i, _j] = 1 + 3 * _i + _j


def f_s2(th):
    st = PMM2DStackHybrid(1.2e-6, n_substrate=1.0, n_superstrate=1.45,
                          degree=7, n_orders=3, formulation="laurent")
    st.add_layer(0.3e-6, eps_cell=C4)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(1.0e-6, theta=th)
    _o, R, T, J = st.solve()
    return s(R, T, J)


def f_s2t(d):
    st = PMM2DStackHybrid(1.2e-6, n_substrate=1.0, n_superstrate=1.45,
                          degree=7, n_orders=3, formulation="laurent")
    st.add_layer(0.3e-6, eps_cell=jnp.asarray(C4).at[0, 0].add(d),
                 region_layout=LAY)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(1.0e-6, theta=0.0)
    _o, R, T, J = st.solve()
    return s(R, T, J)


CASES = {"berreman_exy_0": (f_bm, 0.0), "pmmjones1d_0": (f_pj, 0.0),
         "pmmjones1d_02": (f_pj, 0.2), "pmmstack_shared_2L_0": (f_ps, 0.0),
         "stack2d_c4v_laurent_0": (f_s2, 0.0),
         "stack2d_traced_corner_laurent_0": (f_s2t, 0.0)}


def timed(fn, x):
    x = jnp.asarray(x)
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
for name, (f, x0) in CASES.items():
    rec = {}
    rec["grad_compile_s"], rec["grad_s"] = timed(jax.jit(jax.grad(f)), x0)
    rec["fwd_compile_s"], rec["fwd_s"] = timed(jax.jit(f), x0)
    out[name] = rec
    print(name, {k: round(v, 4) for k, v in rec.items()}, flush=True)
print(dump("h_timing.json", out))
