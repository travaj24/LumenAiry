"""Does the cluster rule's lift cross a branch point of the CONSUMER that is
not at 0?  pmm_jones_1d's half-space geometric eig Kx2 (eigenvalues
mu ~ (kx/k0)^2) feeds q = sqrt(eps_half - mu): non-smooth at mu = eps_half
(a Rayleigh anomaly).  _jax_cluster_routed passes anchor 0 for every problem.
At normal incidence the +-m pair of Kx2 is exactly degenerate; d/dangle
splits it.  Scan the wavelength towards the m=1 substrate anomaly
(lambda = P * n_sub) and compare AD with a Richardson FD (small steps) and
with the FD-free mirror identity dX_{+1} = -dX_{-1}."""
import json
import sys

from _vc import BUILD, fd_rich, jax, jnp, np, premise_ok, rel

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.pmm import pmm_jones_1d

P = 1.2
ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)
NSUB, NSUP, D = 1.0, 1.45, 0.45
out = {"build": BUILD}


def make(wl):
    o = np.asarray(pmm_jones_1d(P, ER, EG, NSUB, NSUP, D, 0.5, wl,
                                degree=12, stabilize=False)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T, _J = pmm_jones_1d(P, ER, EG, NSUB, NSUP, D, 0.5, wl,
                                    angle=t, degree=12, stabilize=False)
        return xp.concatenate([xp.stack([R[p][i] for p in (0, 1)
                                         for i in idx]),
                               xp.stack([T[p][i] for p in (0, 1)
                                         for i in idx])])
    return f


# the Kx2 spectrum scale (for the lift size) via a recording run
def kx2_scale(wl):
    recs = []
    tok = RC._JAX_TWIN_EIG_ROUTE.set(
        lambda L, G, K: (recs.append(np.asarray(L)),
                         RC._jax_twin_eig_plain(L, G))[1])
    try:
        pmm_jones_1d(P, jnp.asarray(ER), EG, NSUB, NSUP, D, 0.5, wl,
                     angle=0.0, degree=12, stabilize=False)
    finally:
        RC._JAX_TWIN_EIG_ROUTE.reset(tok)
    res = []
    for L in recs:
        lam = np.linalg.eigvals(L)
        res.append((L.shape[0], float(np.max(np.abs(lam)))))
    return res


rows = []
for delta in [float(x) for x in sys.argv[1:]] or [3e-2, 1e-2, 3e-3, 1e-3,
                                                    3e-4]:
    wl = P * NSUB * (1.0 + delta)     # m=1 substrate order just evanescent
    f = make(wl)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    from lumenairy.backend import jax_cluster_rule
    with jax_cluster_rule(False):
        goff = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    # FD steps well inside the distance to the anomaly in angle:
    # dkx = n_sup * h must stay << (kx - n_sub) ~ delta
    hs = tuple(min(1e-3, delta * 1e-2) * s for s in (1.0, 0.3, 0.1))
    fd, ratio, expect = fd_rich(lambda t: f(t, np), 0.0, hs=hs)
    mirror = float(np.max(np.abs(g[0::2] + g[1::2])) / np.max(np.abs(g)))
    mirror_fd = float(np.max(np.abs(fd[0::2] + fd[1::2])) / np.max(np.abs(fd)))
    r = dict(delta=delta, wl=wl, err_on=rel(g, fd), err_off=rel(goff, fd),
             mirror_on=mirror, mirror_fd=mirror_fd,
             premise=premise_ok(ratio, expect), ratio_minmax=[
                 float(ratio.min()), float(ratio.max())], expect=expect,
             gmax=float(np.max(np.abs(fd))), hs=hs, scales=kx2_scale(wl))
    print(r, flush=True)
    rows.append(r)
out["rows"] = rows
json.dump(out, open(f"p1_rayleigh_anchor_{BUILD}.json", "w"), indent=1)
