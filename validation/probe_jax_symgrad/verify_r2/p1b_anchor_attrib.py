"""Attribution of p1: (a) pmm_efficiency_1d (whose half-space pencils give
q^2 directly, branch point at 0 = the anchor passed) at the same distances
from the anomaly; (b) pmm_jones_1d with the rule's anchors patched to also
hold eps_sub, eps_sup (the branch points of the Kx2 consumer)."""
import json
import sys

from _vc import BUILD, fd_rich, jax, jnp, np, premise_ok, rel

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d

P, NSUB, NSUP, D = 1.2, 1.0, 1.45, 0.45
ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)
out = {"build": BUILD, "eff1d": [], "jones_patched": []}
deltas = [float(x) for x in sys.argv[1:]] or [1e-3, 3e-4]


def jones_f(wl):
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


def eff_f(wl, pol):
    o = np.asarray(pmm_efficiency_1d(P, 4.0, 1.0, NSUB, NSUP, D, 0.5, wl,
                                     degree=12, stabilize=False)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T = pmm_efficiency_1d(P, xp.asarray(4.0 + 0j), 1.0, NSUB,
                                     NSUP, D, 0.5, wl, angle=t,
                                     polarization=pol, degree=12,
                                     stabilize=False)
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    return f


for delta in deltas:
    wl = P * NSUB * (1.0 + delta)
    hs = tuple(min(1e-3, abs(delta) * 1e-2) * s for s in (1.0, 0.3, 0.1))
    for pol in ("te", "tm"):
        f = eff_f(wl, pol)
        g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
        fd, ratio, ex = fd_rich(lambda t: f(t, np), 0.0, hs=hs)
        r = dict(delta=delta, pol=pol, err=rel(g, fd),
                 mirror=float(np.max(np.abs(g[0::2] + g[1::2]))
                              / np.max(np.abs(g))),
                 premise=premise_ok(ratio, ex))
        print("eff1d", r, flush=True)
        out["eff1d"].append(r)
    orig = RC._jax_eig_cluster_adjoint

    def patched(eig_fn, problems, consumer, **kw):
        n = len(tuple(problems))
        kw["anchors"] = ((0.0, NSUB ** 2, NSUP ** 2),) * n
        return orig(eig_fn, problems, consumer, **kw)
    f = jones_f(wl)
    fd, ratio, ex = fd_rich(lambda t: f(t, np), 0.0, hs=hs)
    g0 = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    RC._jax_eig_cluster_adjoint = patched
    try:
        g1 = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    finally:
        RC._jax_eig_cluster_adjoint = orig
    r = dict(delta=delta, err_as_shipped=rel(g0, fd),
             err_anchor_patched=rel(g1, fd), premise=premise_ok(ratio, ex))
    print("jones", r, flush=True)
    out["jones_patched"].append(r)
json.dump(out, open(f"p1b_anchor_attrib_{BUILD}.json", "w"), indent=1)
