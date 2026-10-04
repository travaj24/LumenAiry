"""V2: pmm_efficiency_1d (JAX) on the verifier's grating (P 0.85, ridge
n 2.3 / groove 1.35, duty 0.4, depth 0.31, n_sup 1, n_sub 1.6, degree 10):
d / d(angle) at 0 and off normal (1e-8 .. 1e-3 rad), d / d(ridge index) at
0 (a SYMMETRY-KEEPING parameter), a lossy ridge; rule ON / OFF; the
FD-free mirror identity at 0; and the PENCIL vs FOLD claim (the rule handed
(B^-1 A, None) instead of (A, B)).
"""
import sys

from _fix import G1, pmm1d_f
from _v import dump, fd_halving, jax, jnp, np, rel

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.pmm import pmm_efficiency_1d

GAP0 = RC._EIG_CLUSTER_GAP_REL
ORIG = RC._jax_eig_cluster_adjoint


def fold_adjoint(eig_fn, problems, consumer, **kw):
    """The rejected first version: the folded operator, Euclidean Gram."""
    probs = tuple((jnp.linalg.solve(B, A), None) for A, B in problems)
    return ORIG(lambda A, _G: eig_fn(A, jnp.eye(A.shape[0], dtype=A.dtype)),
                probs, consumer, **kw)


o = np.asarray(pmm_efficiency_1d(G1["period"], G1["n_ridge"], G1["n_groove"],
                                 G1["n_sub"], G1["n_sup"], G1["depth"],
                                 G1["duty"], G1["wl"], degree=G1["degree"],
                                 stabilize=False)[0])
n_o = o.size
# mirror pairs (+m, -m) inside R and inside T
pairs = [(int(np.nonzero(o == m)[0][0]), int(np.nonzero(o == -m)[0][0]))
         for m in o if m > 0]


def ad(f, x, mode="on"):
    if mode == "off":
        RC._EIG_CLUSTER_GAP_REL = 0.0
    elif mode == "fold":
        RC._jax_eig_cluster_adjoint = fold_adjoint
    try:
        return np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x))
    finally:
        RC._EIG_CLUSTER_GAP_REL = GAP0
        RC._jax_eig_cluster_adjoint = ORIG


out = {"orders": o.tolist()}
only = set(sys.argv[1:])
for pol in ("te", "tm"):
    for which, loss, xs in (("angle", 0.0, (0.0, 1e-8, 1e-6, 1e-5, 1e-4,
                                            1e-3)),
                            ("index", 0.0, (0.0,)),
                            ("angle", 0.05, (0.0, 1e-5))):
        f = pmm1d_f(pol, which, loss)
        for x in xs:
            name = f"{pol}_{which}_loss{loss}_x{x:g}"
            if only and name not in only:
                continue
            fd, prem = fd_halving(lambda t: f(t, np), x, h0=4e-3)
            rec = {"premise": prem, "fd_max": float(np.max(np.abs(fd)))}
            modes = ("on", "off") + (("fold",) if which == "angle" else ())
            for m in modes:
                g = ad(f, x, m)
                rec[f"rel_{m}"] = rel(g, fd)
                rec[f"abs_{m}"] = float(np.max(np.abs(g - fd)))
                if x == 0.0 and which == "angle":
                    mirror = max(max(abs(g[i] + g[j]),
                                     abs(g[n_o + i] + g[n_o + j]))
                                 for i, j in pairs)
                    rec[f"mirror_{m}"] = float(mirror / np.max(np.abs(g)))
            out[name] = rec
            print(name, {k: (f"{v:.3g}" if isinstance(v, float) else v)
                         for k, v in rec.items()}, flush=True)
dump("v2_pmm1d", out)
