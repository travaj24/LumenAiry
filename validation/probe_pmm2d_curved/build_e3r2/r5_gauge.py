"""R5: GAUGE INVARIANCE of the symmetry-breaking gradient.  The eigensolver's
basis inside every exactly degenerate cluster (gap <= 1e-12 max|lam|) is
replaced by a random unitary rotation of it (and every mode gets a random
phase): the forward values must not move beyond round-off and, with the
degenerate-cluster rule, neither must the gradient; with the rule OFF
(the E3 build's adjoint) the gradient moves -- that was V-E3-1.

    python r5_gauge.py M CASE

Writes the max relative change of AD (over R00, T00 E_x, T00 E_y) for two
seeds, rule on and off, plus the forward change.
"""
import sys

from _r2 import P, PMM2DStackPure, dump, jax, jnp, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Ellipse, FilletRect, Rect

M, CASE = int(sys.argv[1]), sys.argv[2]
CASES = {
    "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
    "ellipse_a": (0.33, lambda x: [Ellipse(0.6, 0.6, x, 0.33, 3.5)]),
    "fillet_sq_w": (0.6, lambda x: [FilletRect(0.6, 0.6, x, 0.6, 0.1,
                                               3.5)]),
}
x0, shp = CASES[CASE]


def rotating(orig, seed):
    """``_stag_geneig_jax`` with the basis inside each exact cluster rotated
    by a random unitary (the polar factor of a cluster-masked Gaussian) and a
    random phase on every mode."""
    def eig(L, G, tau_rel=None):
        lam, V = orig(L, G, tau_rel)
        n = lam.shape[0]
        s = jnp.max(jnp.abs(lam))
        K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
            n, dtype=bool)
        rng = np.random.default_rng(seed * 1000 + n)
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = jnp.where(K, X, 0.0)
        w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
        U = Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T)
        return lam, V @ jax.lax.stop_gradient(U)
    return eig


st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=shp(x0), background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()


def f(x):
    p = tw.params()
    p["layers"][0]["shapes"] = shp(x)
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])


orig = JT._stag_geneig_jax
out = {"M": M, "case": CASE}
for rule, gap in (("on", None), ("off", 0.0)):
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    try:
        vals, grads = [], []
        for seed in (None, 1, 2):
            JT._stag_geneig_jax = orig if seed is None else rotating(orig,
                                                                     seed)
            try:
                vals.append(np.asarray(jax.jit(lambda x: f(x))(x0)))
                grads.append(np.asarray(jax.jit(jax.jacrev(lambda x: f(x)))(x0)))
            finally:
                JT._stag_geneig_jax = orig
    finally:
        JT._E3_EIG_CLUSTER_GAP_REL = None
    sc = float(np.max(np.abs(grads[0])))
    rec = {"AD": [g.tolist() for g in grads],
           "grad_change_rel": [float(np.max(np.abs(g - grads[0]))) / sc
                               for g in grads[1:]],
           "value_change": [float(np.max(np.abs(v - vals[0])))
                            for v in vals[1:]]}
    out[f"rule_{rule}"] = rec
    print(CASE, M, "rule", rule, "grad change %.2e %.2e" % tuple(
        rec["grad_change_rel"]), "value change %.1e %.1e" % tuple(
        rec["value_change"]), flush=True)
print(dump(f"r5_gauge_{CASE}_M{M}.json", out))
