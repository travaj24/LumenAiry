"""V1: rcwa_efficiency_2d (JAX eps_cell) AD vs a premise-checked NumPy FD on
the verifier's own C4v cell (24 x 24 px, P 0.9, post eps 5 in 1.8,
n_sup 1, n_sub 1.52, depth 0.37), rule ON (default) and OFF
(_EIG_CLUSTER_GAP_REL = 0, the plain composition = the pre-fix gradient).

Outputs: all R and T (every order).  FD: NumPy, symmetry=False (the full
solve, the same algorithm as the JAX path), central differences at
h = 4e-4, 2e-4, 1e-4, Richardson of the last two; premise = ratio of
successive rung changes (4 for h^2).
"""
import sys

from _fix import rcwa_f
from _v import dump, fd_halving, jax, jnp, np, rel, tic

import lumenairy.elements.rcwa._core as RC

GAP0 = RC._EIG_CLUSTER_GAP_REL
CASES = [
    # name, direction, pol, n_orders, kwargs
    ("x_te_n2", "x", "te", 2, {}),
    ("x_tm_n2", "x", "tm", 2, {}),
    ("x_te_n3", "x", "te", 3, {}),
    ("x_tm_n3", "x", "tm", 3, {}),
    ("corner_te_n2", "corner", "te", 2, {}),
    ("corner_tm_n2", "corner", "tm", 2, {}),
    ("diag_te_n2", "diag", "te", 2, {}),
    ("keep_te_n2", "keep", "te", 2, {}),
    ("keep_tm_n2", "keep", "tm", 2, {}),
    ("lossy_x_te_n2", "x", "te", 2, {"loss": 0.8}),
    ("lossy_x_tm_n2", "x", "tm", 2, {"loss": 0.8}),
    ("li_x_te_n2", "x", "te", 2, {"formulation": "li"}),
    ("li_x_tm_n2", "x", "tm", 2, {"formulation": "li"}),
    ("conical_corner_te_n2", "corner", "te", 2, {"theta": 0.2, "phi": 0.3}),
    ("conical_corner_tm_n2", "corner", "tm", 2, {"theta": 0.2, "phi": 0.3}),
    ("oblique_x_tm_n2", "x", "tm", 2, {"theta": 0.15}),
]
only = set(sys.argv[1:])
out = {}
for name, d, pol, n, kw in CASES:
    if only and name not in only:
        continue
    f = rcwa_f(d, pol, n_orders=n, **kw)
    t0 = tic()
    fd, prem = fd_halving(lambda t: f(t, np), 0.0)
    rec = {"premise": prem, "fd_max": float(np.max(np.abs(fd))),
           "t_fd": tic() - t0}
    for rule, gap in (("on", GAP0), ("off", 0.0)):
        RC._EIG_CLUSTER_GAP_REL = gap
        try:
            t0 = tic()
            g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
            rec[f"t_{rule}"] = tic() - t0
        finally:
            RC._EIG_CLUSTER_GAP_REL = GAP0
        rec[f"rel_{rule}"] = rel(g, fd)
        rec[f"abs_{rule}"] = float(np.max(np.abs(g - fd)))
    out[name] = rec
    print(name, {k: (f"{v:.3g}" if isinstance(v, float) else v)
                 for k, v in rec.items()}, flush=True)
dump("v1_rcwa" + ("_" + "_".join(sorted(only)) if only else ""), out)
