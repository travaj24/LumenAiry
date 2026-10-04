"""M1 (round 3, verifier P1-1): d / d(angle) at exactly normal incidence
near a Rayleigh anomaly, as the distance to it shrinks -- pmm_jones_1d
(P 1.2, ridge 4 / groove 1, n_sub 1, n_sup 1.45, degree 12; the m = +-1
substrate orders graze at wl = 1.2) and PMMStack (the verifier's p10b
fixture; graze at wl = 1 um), wl = wl_c (1 + delta).

Per delta: the FD-free MIRROR identity of AD (|dX_{+1} + dX_{-1}| /
max|g|) with the shipped per-problem anchors, with every anchor forced to
0 (round 2), and with the rule OFF; the same identity of the FD; AD vs a
Richardson FD whose steps scale with delta (h = delta x (1e-1, 3e-2, 1e-2)
rad, so the FD never crosses the anomaly), and that FD's h^2 premise.

    python m1_anomaly.py [jones|stack|pmm1d]

pmm1d: pmm_efficiency_1d TE on the same grating (its half-space eigs are
pencils whose eigenvalue IS q^2, so the default anchor 0 is the anomaly).
"""
import sys

from _h import dump, jax, jnp, np

import lumenairy.elements.rcwa._core as RC
from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.pmm import PMMStack, pmm_jones_1d

WHICH = sys.argv[1] if len(sys.argv) > 1 else "jones"
ER = 4.0 * np.eye(3, dtype=complex)


def make(delta):
    if WHICH == "pmm1d":
        from lumenairy.elements.pmm import pmm_efficiency_1d
        wl = 1.2 * (1.0 + delta)
        o = np.asarray(pmm_efficiency_1d(1.2, 2.0, 1.0, 1.0, 1.45, 0.45, 0.5,
                                         wl, degree=12, stabilize=False)[0])
        idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

        def f(t, xp):
            _o, R, T = pmm_efficiency_1d(1.2, xp.asarray(2.0 + 0j), 1.0, 1.0,
                                         1.45, 0.45, 0.5, wl, angle=t,
                                         degree=12, stabilize=False)
            return xp.concatenate([xp.stack([R[i] for i in idx]),
                                   xp.stack([T[i] for i in idx])])
        return f
    if WHICH == "jones":
        wl = 1.2 * (1.0 + delta)
        o = np.asarray(pmm_jones_1d(1.2, ER, np.eye(3, dtype=complex), 1.0,
                                    1.45, 0.45, 0.5, wl, degree=12,
                                    stabilize=False)[0])
        idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

        def f(t, xp):
            _o, R, T, _J = pmm_jones_1d(1.2, ER, np.eye(3, dtype=complex),
                                        1.0, 1.45, 0.45, 0.5, wl, angle=t,
                                        degree=12, stabilize=False)
            return xp.concatenate([
                xp.stack([R[p][i] for p in (0, 1) for i in idx]),
                xp.stack([T[p][i] for p in (0, 1) for i in idx])])
        return f
    wl = 1.0e-6 * (1.0 + delta)

    def solve(t):
        st = PMMStack(1.0e-6, n_substrate=1.0, n_superstrate=1.45, degree=10)
        st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                       (0.15, 3.0), (0.5, 1.5)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
        st.set_source(wl, angle=t)
        return st.solve()
    o = np.asarray(solve(0.0)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T, _J = solve(t)
        return xp.concatenate([
            xp.stack([R[p][i] for p in (0, 1) for i in idx]),
            xp.stack([T[p][i] for p in (0, 1) for i in idx])])
    return f


def mirror(g):
    g = np.asarray(g)
    return float(np.max(np.abs(g[0::2] + g[1::2])) / np.max(np.abs(g)))


_orig = RC._jax_twin_eig


def anchor0(L, G=None, K=None, anchors=(0.0,)):
    return _orig(L, G, K, (0.0,))


out = {}
for delta in (1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
    f = make(delta)
    rec = {}
    for name in ("anchors", "anchor0", "off"):
        if name == "anchor0":
            RC._jax_twin_eig = anchor0
        try:
            with jax_cluster_rule(name != "off"):
                g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
        finally:
            RC._jax_twin_eig = _orig
        rec[name] = {"AD": g.tolist(), "mirror": mirror(g)}
    hs = [delta * s for s in (1e-1, 3e-2, 1e-2)]
    rows = [(np.asarray(f(h, np)) - np.asarray(f(-h, np))) / (2 * h)
            for h in hs]
    fd = (9.0 * rows[2] - rows[1]) / 8.0
    c1, c2 = np.abs(rows[0] - rows[1]), np.abs(rows[1] - rows[2])
    big = np.abs(fd) >= 1e-3 * np.max(np.abs(fd))
    rec["FD"] = {"value": fd.tolist(), "mirror": mirror(fd),
                 "premise": (c1[big] / np.maximum(c2[big], 1e-300)).round(2).tolist(),
                 "resolution": float(np.finfo(float).eps * np.max(np.abs(
                     np.asarray(f(0.0, np)))) / hs[2])}
    for name in ("anchors", "anchor0", "off"):
        g = np.asarray(rec[name]["AD"])
        rec[name]["rel_vs_FD"] = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    out[repr(delta)] = rec
    print(WHICH, "delta %.0e" % delta,
          " ".join("%s: mirror %.1e rel %.1e" % (k, rec[k]["mirror"], rec[k]["rel_vs_FD"])
                   for k in ("anchors", "anchor0", "off")),
          "| FD mirror %.1e res %.1e premise %s" % (
              rec["FD"]["mirror"], rec["FD"]["resolution"], rec["FD"]["premise"][:3]),
          flush=True)
print(dump(f"m1_anomaly_{WHICH}.json", out))
