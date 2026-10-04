"""N1 (round 3, verifier P1-2): RCWAStack (JAX) with an EXACTLY uniform
scalar eps_cell layer (eps 2.25) over a cross layer (eps 4), differentiated
in a zero-mean random pattern direction t -- the start of a topology
optimisation; the verifier's fixture.  AD at t = 0 vs the FD at 0 and vs
the mean of AD(+-1e-6), at normal and conical (0.2, 0.3) incidence; plus
the forward bytes at t = 0 (jit, eager) to show the analytic shortcut's
value is kept.  PRE (5ea82b44) vs r3.

    python n1_uniform_layer.py
"""
import hashlib

from _h import dump, jax, jnp, np

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.rcwa import RCWAStack

S = 24
CROSS = np.zeros((S, S), bool)
CROSS[8:16, :] = True
CROSS[:, 8:16] = True
RND = np.random.default_rng(11).standard_normal((S, S))
RND -= RND.mean()


def make(theta, phi):
    def f(t, xp):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.25, eps_cell=xp.asarray(np.full((S, S), 2.25 + 0j))
                     + t * xp.asarray(RND))
        st.add_layer(0.2, eps_cell=xp.asarray(np.where(CROSS, 4.0, 1.0)
                                              .astype(complex)))
        st.set_source(1.0, theta=theta, phi=phi)
        _o, R, T = st.solve().efficiencies()
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    return f


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


out = {}
for name, (th, ph) in (("normal", (0.0, 0.0)), ("conical", (0.2, 0.3))):
    f = make(th, ph)
    RC._clear_rcwa_caches()
    vals = []
    for x in (0.0, 1e-6, -1e-6):
        RC._clear_rcwa_caches()   # the PRE tree leaks tracers through it
        vals.append(np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x)))
    g0, gp, gm = vals
    hs = (1e-3, 3e-4, 1e-4)
    rows = [(np.asarray(f(h, np)) - np.asarray(f(-h, np))) / (2 * h)
            for h in hs]
    fd = (9.0 * rows[2] - rows[1]) / 8.0
    RC._clear_rcwa_caches()
    v_jit = np.asarray(jax.jit(lambda t: f(t, jnp))(0.0))
    RC._clear_rcwa_caches()
    v_eager = np.asarray(f(jnp.asarray(0.0), jnp))
    rec = {"AD0_vs_FD": rel(g0, fd), "AD0_vs_near": rel(g0, 0.5 * (gp + gm)),
           "near_vs_FD": rel(0.5 * (gp + gm), fd),
           "sha_jit": hashlib.sha256(v_jit.tobytes()).hexdigest(),
           "sha_eager": hashlib.sha256(v_eager.tobytes()).hexdigest(),
           "jit_vs_numpy": float(np.max(np.abs(v_jit - np.asarray(f(0.0, np)))))}
    out[name] = rec
    print(name, {k: (v if isinstance(v, str) else "%.2e" % v)
                 for k, v in rec.items()}, flush=True)
print(dump("n1_uniform_layer.json", out))
