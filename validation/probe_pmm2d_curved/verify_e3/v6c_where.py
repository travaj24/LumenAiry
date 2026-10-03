"""V6c (V-E3-1): WHICH eig carries the symmetry-breaking gradient error --
the homogeneous (geometric) pencil, whose plane-wave modes are massively
degenerate, or the patterned layer's pencil?  The square pillar's d / d w at
M = 3, with a deterministic degeneracy SPLIT (1e-8 relative, the prototype
fix) applied to only one of the two eig calls of the twin's solve.

    python v6c_where.py M
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Rect

M = int(sys.argv[1])
orig = JT._stag_geneig_jax
SPLIT = 1e-8


def split_eig(L, G, tau_rel=None):
    A = jnp.linalg.solve(G, L)
    n = A.shape[0]
    rng = np.random.default_rng(20261003 + n)
    Rm = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Rm = Rm / np.linalg.norm(Rm, 2)
    sc = jax.lax.stop_gradient(jnp.max(jnp.abs(A)))
    from lumenairy.elements.rcwa import _jax_eig_stable
    return _jax_eig_stable()(A + (SPLIT * sc) * Rm)


st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=[Rect(0.6, 0.6, 0.5, 0.5, 3.5)],
             background_eps=1.0)
st.set_source(WL)
tw = st.jax_twin()


def f(w):
    p = tw.params()
    p["layers"][0]["shapes"] = [Rect(0.6, 0.6, w, 0.5, 3.5)]
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, 12], T[0, 12], T[1, 12]])


def fn(w):
    s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                       n_modes=M, n_orders=2)
    s.add_layer(0.45, shapes=[Rect(0.6, 0.6, w, 0.5, 3.5)],
                background_eps=1.0)
    s.set_source(WL)
    o, R, T, J = s.solve()
    return np.array([R[0, 12], T[0, 12], T[1, 12]])


_r, fd, _c, _rat = ladder(fn, 0.5, [1e-3, 3e-4, 1e-4], P)
sc = float(np.max(np.abs(fd)))
out = {"M": M, "FD_numpy": fd.tolist()}
for tag, which in (("none", ()), ("geometric_only", (0,)),
                   ("layer_only", (1,)), ("both", (0, 1))):
    calls = {"k": 0}

    def patched(L, G, tau_rel=None, which=which, calls=calls):
        k = calls["k"]
        calls["k"] += 1
        return (split_eig if k in which else orig)(L, G, tau_rel)
    JT._stag_geneig_jax = patched
    try:
        g = np.asarray(jax.jit(jax.jacrev(lambda w: f(w)))(0.5))
    finally:
        JT._stag_geneig_jax = orig
    out[tag] = {"AD": g.tolist(), "calls_traced": calls["k"],
                "AD_vs_FDnumpy_rel": float(np.max(np.abs(g - fd))) / sc}
    print(tag, out[tag]["calls_traced"], out[tag]["AD_vs_FDnumpy_rel"],
          flush=True)
print(dump(f"v6c_where_M{M}.json", out))
