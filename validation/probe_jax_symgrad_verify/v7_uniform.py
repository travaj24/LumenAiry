"""V7: a UNIFORM cell differentiated toward a pattern (eps 2.25 + t x post,
t = 0 traced): the structured eig the rule sees is the uniform medium's,
with exact multi-member clusters (TE / TM per order and the (+-m, +-n)
multiplets), while the consumer selects the analytic uniform modes.  AD
(rule on / off) vs FD; finiteness."""
from _fix import post, rcwa_f
from _v import dump, fd_halving, jax, jnp, np, rel

import lumenairy.elements.rcwa._core as RC

GAP0 = RC._EIG_CLUSTER_GAP_REL
out = {}
for pol, th in (("te", 0.0), ("tm", 0.0), ("te", 0.2)):
    f = rcwa_f(post, pol, n_orders=2, base=np.full((24, 24), 2.25),
               theta=th)
    fd, prem = fd_halving(lambda t: f(t, np), 0.0)
    rec = {"premise": prem, "fd_max": float(np.max(np.abs(fd)))}
    for rule, gap in (("on", GAP0), ("off", 0.0)):
        RC._EIG_CLUSTER_GAP_REL = gap
        try:
            g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
        finally:
            RC._EIG_CLUSTER_GAP_REL = GAP0
        rec[f"finite_{rule}"] = bool(np.all(np.isfinite(g)))
        rec[f"rel_{rule}"] = rel(g, fd)
    out[f"{pol}_theta{th}"] = rec
    print(pol, th, rec, flush=True)
dump("v7_uniform", out)
