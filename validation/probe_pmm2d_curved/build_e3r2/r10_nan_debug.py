"""R10: the fillet far from its reference (frozen at r = 0.05, evaluated at
r = 0.00192, the verifier's V-E3-3 control point): which part of the
cluster rule turns the gradient NaN?  Captures the eig problems through the
rule's entry and inspects the lift ingredients.

    python r10_nan_debug.py
"""
from _r2 import P, PMM2DStackPure, jax, jnp, np

import lumenairy.elements.rcwa._core as RCC
from lumenairy.elements.pmm import FilletRect

st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2, backend="jax")
st.add_layer(0.4, shapes=[FilletRect(0.6, 0.6, 0.6, 0.5, 0.05, 4.0)],
             background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()
cap = {}
orig = RCC._jax_eig_cluster_adjoint


def spy(eig_fn, problems, consumer, **kw):
    cap["problems"] = [(np.asarray(L), np.asarray(G)) for L, G in problems]
    cap["anchors"] = kw.get("anchors")
    return orig(eig_fn, problems, consumer, **kw)


RCC._jax_eig_cluster_adjoint = spy


def f(r):
    p = tw.params()
    p["layers"][0]["shapes"] = [FilletRect(0.6, 0.6, 0.6, 0.5, r, 4.0)]
    return st.solve(params=p)[2][0, tw.p0]


r = jnp.asarray(0.00192)
print("value", float(f(r)))
for k, ((L, G), an) in enumerate(zip(cap["problems"], cap["anchors"])):
    A = np.linalg.solve(G, L)
    lam, V = np.linalg.eig(A)
    s = np.max(np.abs(lam))
    D = np.abs(lam[:, None] - lam[None, :])
    close = (D <= 1e-6 * s) & ~np.eye(lam.size, dtype=bool)
    mem = close.any(1)
    K = close | (np.eye(lam.size, dtype=bool) & mem[:, None])
    trans = bool(np.all(((K.astype(int) @ K.astype(int)) > 0) == K))
    SK = np.where(K, V.conj().T @ G @ V, 0) + np.diag(np.where(mem, 0, 1.0))
    w = np.linalg.eigvalsh(SK)
    dN, anyc = RCC._eig_cluster_lift(jnp.asarray(lam), jnp.asarray(V),
                                     jnp.asarray(G), 1e-6, 1e-7, an)
    print(k, "n", lam.size, "members", int(mem.sum()), "transitive", trans,
          "SK min eig %.2e" % w.min(), "lift finite",
          bool(np.all(np.isfinite(np.asarray(dN)))),
          "min|lam|/s %.1e" % (np.min(np.abs(lam)) / s))
for gap in (0.0, None):
    RCC._jax_eig_cluster_adjoint = orig
    import lumenairy.elements.pmm._jax_twod_staggered as JT
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    g = jax.jit(jax.grad(lambda x: f(x)))(0.00192)
    JT._E3_EIG_CLUSTER_GAP_REL = None
    print("gap", gap, "grad", float(g))

# the twin's own FD at the point (h / P = 1e-4 .. 1e-5: the fillet radius is
# 1.6e-3 P here, so the ladder stays well inside it)
from _r2 import dump, ladder  # noqa: E402

RCC._jax_eig_cluster_adjoint = orig
_rows, fd, _ch, rat = ladder(jax.jit(lambda x: f(x)), 0.00192,
                             [1e-4, 3e-5, 1e-5], P)
print("FD", float(fd), "premise", np.asarray(rat).ravel())
out = {"FD_twin": float(fd), "premise": np.asarray(rat).ravel().tolist()}
for gap in (0.0, None):
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    g = float(jax.jit(jax.grad(lambda x: f(x)))(0.00192))
    JT._E3_EIG_CLUSTER_GAP_REL = None
    out[f"gap={gap}"] = {"AD": g, "rel": abs(g - float(fd)) / abs(float(fd))}
    print("gap", gap, "rel err %.2e" % out[f"gap={gap}"]["rel"])
print(dump("r10_fillet_far_from_reference.json", out))
